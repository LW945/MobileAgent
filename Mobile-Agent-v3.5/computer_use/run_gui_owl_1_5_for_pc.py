"""
Usage:
    cd Mobile-Agent-v3.5/computer_use
    python run_gui_owl_1_5_for_pc.py \
        --api_key "Your API key" \
        --base_url "Your base url of vllm service" \
        --instruction "The instruction you want the agent to complete" \
        --model "Model name" \
        --add_info "Optional supplementary knowledge"
"""

import argparse
from collections import defaultdict
from datetime import datetime
import json
import os
import time

from PIL import Image

from utils import (
    ComputerTools,
    StepPopup,
    annotate_screenshot,
    build_messages,
    extract_tool_calls,
    get_output_dir,
    sanitize_filename,
    smart_resize,
    GUIOwlWrapper
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Computer-Agent-v3.5: Desktop GUI automation agent"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="DashScope API key",
    )
    parser.add_argument("--base_url", type=str, required=True,
                        help="Base URL for the VLM service.")
    parser.add_argument(
        "--instruction",
        type=str,
        required=True,
        help="The task instruction for the agent to complete",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Model name for the VLM service",
    )
    parser.add_argument(
        "--add_info",
        type=str,
        default="",
        help="Optional supplementary knowledge for the task",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=50,
        help="Maximum number of interaction steps (default: 50)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="off",
        choices=("off", "basic", "detailed"),
        help="Enable task profiling logs and summaries.",
    )
    return parser.parse_args()


def rescale_coordinates(action_parameter, resized_width, resized_height):
    """
    Convert normalized coordinates (0–1000 range) to actual pixel
    coordinates based on the resized image dimensions.
    """
    for key in ("coordinate", "coordinate1", "coordinate2"):
        if key in action_parameter:
            action_parameter[key][0] = int(
                action_parameter[key][0] / 1000 * resized_width
            )
            action_parameter[key][1] = int(
                action_parameter[key][1] / 1000 * resized_height
            )


def execute_action(computer_tools, action_parameter):
    """
    Execute a single action on the desktop.

    Returns:
        stop (bool): True if the agent wants to terminate.
    """
    action_type = action_parameter["action"]

    if action_type in ("click", "left_click"):
        computer_tools.left_click(
            action_parameter["coordinate"][0],
            action_parameter["coordinate"][1],
        )

    elif action_type == "mouse_move":
        computer_tools.mouse_move(
            action_parameter["coordinate"][0],
            action_parameter["coordinate"][1],
        )

    elif action_type == "middle_click":
        computer_tools.middle_click(
            action_parameter["coordinate"][0],
            action_parameter["coordinate"][1],
        )

    elif action_type in ("right click", "right_click"):
        computer_tools.right_click(
            action_parameter["coordinate"][0],
            action_parameter["coordinate"][1],
        )

    elif action_type == "open app":
        computer_tools.open_app(action_parameter["app_name"])

    elif action_type in ("key", "hotkey"):
        computer_tools.press_key(action_parameter["keys"])

    elif action_type == "type":
        computer_tools.type(action_parameter["text"])

    elif action_type in ("drag", "left_click_drag"):
        computer_tools.left_click_drag(
            action_parameter["coordinate"][0],
            action_parameter["coordinate"][1],
        )

    elif action_type == "scroll":
        if "coordinate" in action_parameter:
            computer_tools.mouse_move(
                action_parameter["coordinate"][0],
                action_parameter["coordinate"][1],
            )
        computer_tools.scroll(action_parameter.get("pixels", 1))

    elif action_type == "hscroll":
        if "coordinate" in action_parameter:
            computer_tools.mouse_move(
                action_parameter["coordinate"][0],
                action_parameter["coordinate"][1],
            )
        computer_tools.hscroll(action_parameter.get("pixels", 1))

    elif action_type in ("computer_double_click", "double_click"):
        computer_tools.double_click(
            action_parameter["coordinate"][0],
            action_parameter["coordinate"][1],
        )

    elif action_type == "triple_click":
        computer_tools.triple_click(
            action_parameter["coordinate"][0],
            action_parameter["coordinate"][1],
        )

    elif action_type == "call_user":
        StepPopup.show_blocking(
            "User Interaction Required",
            "Please perform the requested manual operation.",
            image_path="",
            timeout_sec=120,
            width=960,
            height=540,
        )
        print("Manual action completed, resuming...")

    elif action_type == "wait":
        time.sleep(action_parameter.get("time", 2))

    elif action_type == "answer":
        StepPopup.show_blocking(
            "Task Finished",
            action_parameter["text"],
            image_path="",
            timeout_sec=120,
            width=960,
            height=540,
        )
        return True  # signal to stop

    elif action_type in ("stop", "terminate", "done"):
        status = action_parameter.get("status", "success")
        StepPopup.show_blocking(
            "Task Completed",
            f"Task completed with status: {status}",
            image_path="",
            timeout_sec=120,
            width=960,
            height=540,
        )
        return True  # signal to stop

    elif action_type == "interact":
        StepPopup.show_blocking(
            "User Interaction Required",
            action_parameter.get("text", "Please interact with the dialog."),
            image_path="",
            timeout_sec=120,
            width=960,
            height=540,
        )
        print("User interaction completed, resuming...")

    else:
        raise ValueError(f"Unsupported action type: {action_type}")

    return False  # continue execution


def extract_reasoning_content(raw_response):
    """Best-effort extraction of reasoning text from an OpenAI-compatible response."""
    if raw_response is None:
        return None

    try:
        message = raw_response.choices[0].message
    except (AttributeError, IndexError, KeyError, TypeError):
        return None

    thought = getattr(message, "reasoning_content", None)
    if thought:
        return thought

    reasoning = getattr(message, "reasoning", None)
    if isinstance(reasoning, str) and reasoning:
        return reasoning

    if isinstance(reasoning, list):
        parts = []
        for item in reasoning:
            if isinstance(item, dict):
                text = item.get("text")
            else:
                text = getattr(item, "text", None)
            if text:
                parts.append(text)
        if parts:
            return "\n".join(parts)

    return None


def current_timestamp():
    """Return a local timezone-aware timestamp suitable for logs."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def append_task_log(log_path, record):
    """Append one task record as JSONL."""
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json_file(path, record):
    """Write one JSON document."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def rounded_seconds(duration):
    """Normalize durations for logs."""
    return round(duration, 3)


def safe_percentage(value, total):
    """Return a rounded percentage without dividing by zero."""
    if total <= 0:
        return 0.0
    return round(value / total * 100, 1)


def blocking_action_types():
    """Action types that intentionally spend wall-clock time waiting."""
    return {"wait", "call_user", "interact", "answer", "stop", "terminate", "done"}


def emit_vllm_request_log(
    log_path,
    instruction,
    event,
    overall_request_stats,
    step_request_stats,
):
    """Persist, print, and aggregate one structured vLLM request event."""
    record = {
        "instruction": instruction,
        "log_path": log_path,
        **event,
    }
    append_task_log(log_path, record)

    context = record.get("context") or {}
    step_id = context.get("step_id")

    if record["phase"] == "send":
        overall_request_stats["send_count"] += 1
        overall_request_stats["payload_prepare_seconds"] += record.get(
            "payload_prepare_seconds", 0.0
        )
        if step_id is not None:
            step_request_stats[step_id]["send_count"] += 1
            step_request_stats[step_id]["payload_prepare_seconds"] += record.get(
                "payload_prepare_seconds", 0.0
            )
    elif record["phase"] == "receive":
        overall_request_stats["receive_count"] += 1
        overall_request_stats["network_latency_seconds"] += record.get(
            "latency_seconds", 0.0
        )
        if record.get("status") == "error":
            overall_request_stats["error_count"] += 1
        if step_id is not None:
            step_request_stats[step_id]["receive_count"] += 1
            step_request_stats[step_id]["network_latency_seconds"] += record.get(
                "latency_seconds", 0.0
            )
            if record.get("status") == "error":
                step_request_stats[step_id]["error_count"] += 1

    prefix = f"[VLLM {record['phase'].upper()}]"
    summary = (
        f"{prefix} t={record['timestamp']} req={record['request_index']} "
        f"attempt={record['attempt']}"
    )
    if record["phase"] == "send":
        summary += (
            f" messages={record.get('message_count')} "
            f"images={record.get('image_count')} "
            f"prepare={record.get('payload_prepare_seconds')}s"
        )
    else:
        summary += (
            f" status={record.get('status')} "
            f"latency={record.get('latency_seconds')}s"
        )
        response_id = record.get("response_id")
        if response_id:
            summary += f" response_id={response_id}"
        if record.get("will_retry") is not None:
            summary += f" will_retry={record['will_retry']}"
    if step_id is not None:
        summary += f" step={step_id}"
    print(summary)
    if record.get("error"):
        print(f"[VLLM ERROR] {record['error']}")


def main():
    args = parse_args()

    # Initialize tools
    computer_tools = ComputerTools()
    computer_tools.reset()

    # Prepare output directory
    output_dir = get_output_dir()
    safe_instruction = sanitize_filename(args.instruction)
    task_log_path = os.path.join(output_dir, "task_runs.jsonl")
    request_log_path = os.path.join(output_dir, "vllm_requests.jsonl")
    step_profile_path = os.path.join(output_dir, "step_profile.jsonl")
    action_profile_path = os.path.join(output_dir, "action_profile.jsonl")
    task_summary_path = os.path.join(output_dir, "task_summary.json")

    profile_enabled = args.profile != "off"
    detailed_profile = args.profile == "detailed"

    overall_request_stats = defaultdict(float)
    step_request_stats = defaultdict(lambda: defaultdict(float))
    vllm = GUIOwlWrapper(
        args.api_key,
        args.base_url,
        args.model,
        request_logger=lambda event: emit_vllm_request_log(
            request_log_path,
            args.instruction,
            event,
            overall_request_stats,
            step_request_stats,
        ),
    )

    history = []
    stop_flag = False
    task_status = "running"
    error_message = None
    task_start_time = current_timestamp()
    task_start_monotonic = time.monotonic()
    phase_totals = defaultdict(float)
    action_totals = defaultdict(float)
    slowest_step = None
    slowest_action = None
    completed_steps = 0

    print(f"[TASK START] {task_start_time}")
    print(f"[TASK LOG] {task_log_path}")
    print(f"[VLLM REQUEST LOG] {request_log_path}")
    if profile_enabled:
        print(f"[PROFILE MODE] {args.profile}")
        print(f"[STEP PROFILE LOG] {step_profile_path}")
        if detailed_profile:
            print(f"[ACTION PROFILE LOG] {action_profile_path}")
        print(f"[TASK SUMMARY] {task_summary_path}")

    try:
        for step_id in range(args.max_steps):
            if stop_flag:
                break

            print(f"\nSTEP {step_id}:\n{'=' * 50}")
            step_started_at = time.monotonic()
            step_timestamp = current_timestamp()
            step_phases = defaultdict(float)
            step_action_count = 0
            step_status = "running"

            # Capture screenshot
            screen_shot = os.path.join(output_dir, f"{safe_instruction}_{step_id}.png")
            phase_started_at = time.monotonic()
            screenshot_ok = computer_tools.get_screenshot(screen_shot)
            step_phases["screenshot"] += time.monotonic() - phase_started_at
            if not screenshot_ok:
                print(f"[ERROR] Failed to capture screenshot at step {step_id}")
                step_status = "screenshot_failed"
                step_elapsed_seconds = rounded_seconds(
                    time.monotonic() - step_started_at
                )
                step_phases["other"] += max(
                    0.0,
                    step_elapsed_seconds - rounded_seconds(step_phases["screenshot"]),
                )
                for key, value in step_phases.items():
                    phase_totals[key] += value
                if profile_enabled:
                    step_record = {
                        "instruction": args.instruction,
                        "step_id": step_id,
                        "timestamp": step_timestamp,
                        "status": step_status,
                        "step_elapsed_seconds": step_elapsed_seconds,
                        "phase_durations_seconds": {
                            key: rounded_seconds(value)
                            for key, value in step_phases.items()
                        },
                        "request_stats": dict(step_request_stats.get(step_id, {})),
                        "action_count": step_action_count,
                    }
                    append_task_log(step_profile_path, step_record)
                    if slowest_step is None or step_elapsed_seconds > slowest_step["duration_seconds"]:
                        slowest_step = {
                            "step_id": step_id,
                            "status": step_status,
                            "duration_seconds": step_elapsed_seconds,
                            "dominant_phase": "screenshot",
                        }
                continue

            # Build messages and call the VLM
            phase_started_at = time.monotonic()
            messages = build_messages(
                screen_shot, args.instruction, history, args.model
            )
            step_phases["build_messages"] += time.monotonic() - phase_started_at

            phase_started_at = time.monotonic()
            output_text, _, raw_response = vllm.predict_mm(
                messages,
                request_context={"step_id": step_id},
            )
            step_phases["vllm_request"] += time.monotonic() - phase_started_at

            # Prepend reasoning content if present
            phase_started_at = time.monotonic()
            thought = extract_reasoning_content(raw_response)
            if thought:
                output_text = f"<thinking>\n{thought}\n</thinking>{output_text}"
            step_phases["response_parse"] += time.monotonic() - phase_started_at

            print(output_text)

            # Extract and execute tool calls
            phase_started_at = time.monotonic()
            action_list = extract_tool_calls(output_text)
            step_phases["tool_extract"] += time.monotonic() - phase_started_at

            phase_started_at = time.monotonic()
            dummy_image = Image.open(screen_shot)
            resized_height, resized_width = smart_resize(
                dummy_image.height,
                dummy_image.width,
                factor=16,
                min_pixels=3136,
                max_pixels=1003520 * 200,
            )
            step_phases["image_preprocess"] += time.monotonic() - phase_started_at

            for action_id, action in enumerate(action_list):
                action_parameter = action["arguments"]
                action_type = action_parameter["action"]
                step_action_count += 1

                # Rescale normalized coordinates to actual pixels
                rescale_coordinates(action_parameter, resized_width, resized_height)

                # Execute the action
                action_started_at = time.monotonic()
                should_stop = execute_action(computer_tools, action_parameter)
                action_duration = time.monotonic() - action_started_at
                step_phases["action_execute"] += action_duration
                action_totals[action_type] += action_duration
                if action_type in blocking_action_types():
                    step_phases["blocking_action"] += action_duration

                if detailed_profile:
                    action_record = {
                        "instruction": args.instruction,
                        "step_id": step_id,
                        "action_id": action_id,
                        "timestamp": current_timestamp(),
                        "action_type": action_type,
                        "duration_seconds": rounded_seconds(action_duration),
                        "is_blocking": action_type in blocking_action_types(),
                        "keys": sorted(action_parameter.keys()),
                    }
                    append_task_log(action_profile_path, action_record)

                if slowest_action is None or action_duration > slowest_action["duration_seconds"]:
                    slowest_action = {
                        "step_id": step_id,
                        "action_id": action_id,
                        "action_type": action_type,
                        "duration_seconds": rounded_seconds(action_duration),
                    }

                if should_stop:
                    stop_flag = True
                    task_status = "completed"
                    step_status = "completed"
                    break

                # Annotate screenshot for debugging / visualization
                annotation_started_at = time.monotonic()
                annotate_screenshot(
                    screen_shot,
                    action_parameter,
                    os.path.join(
                        output_dir,
                        f"anno_{safe_instruction}_{step_id}_{action_id}.png",
                    ),
                )
                step_phases["annotation"] += time.monotonic() - annotation_started_at

            if step_status == "running":
                step_status = "ok"

            # Record history
            history.append({"output": output_text, "image": screen_shot})
            completed_steps += 1

            if not stop_flag:
                sleep_started_at = time.monotonic()
                time.sleep(2)
                step_phases["sleep"] += time.monotonic() - sleep_started_at

            step_elapsed = time.monotonic() - step_started_at
            known_step_seconds = sum(
                value
                for key, value in step_phases.items()
                if key != "blocking_action"
            )
            step_phases["other"] += max(0.0, step_elapsed - known_step_seconds)

            if profile_enabled:
                rounded_step_phases = {
                    key: rounded_seconds(value)
                    for key, value in step_phases.items()
                }
                dominant_phase_name, dominant_phase_seconds = max(
                    (
                        (key, value)
                        for key, value in rounded_step_phases.items()
                        if key != "blocking_action"
                    ),
                    key=lambda item: item[1],
                    default=(None, 0.0),
                )
                request_stats_for_step = {
                    key: rounded_seconds(value)
                    for key, value in step_request_stats.get(step_id, {}).items()
                }
                step_record = {
                    "instruction": args.instruction,
                    "step_id": step_id,
                    "timestamp": step_timestamp,
                    "status": step_status,
                    "step_elapsed_seconds": rounded_seconds(step_elapsed),
                    "phase_durations_seconds": rounded_step_phases,
                    "dominant_phase": dominant_phase_name,
                    "dominant_phase_seconds": dominant_phase_seconds,
                    "request_stats": request_stats_for_step,
                    "action_count": step_action_count,
                }
                append_task_log(step_profile_path, step_record)
                if slowest_step is None or step_elapsed > slowest_step["duration_seconds"]:
                    slowest_step = {
                        "step_id": step_id,
                        "status": step_status,
                        "duration_seconds": rounded_seconds(step_elapsed),
                        "dominant_phase": dominant_phase_name,
                    }

            for key, value in step_phases.items():
                phase_totals[key] += value

        if not stop_flag:
            task_status = "max_steps_reached"
            print(f"\n[INFO] Reached maximum steps ({args.max_steps}). Stopping.")

    except Exception as exc:
        task_status = "failed"
        error_message = repr(exc)
        raise
    finally:
        task_end_time = current_timestamp()
        elapsed_seconds = round(time.monotonic() - task_start_monotonic, 3)
        rounded_phase_totals = {
            key: rounded_seconds(value)
            for key, value in phase_totals.items()
        }
        phase_percentages = {
            key: safe_percentage(value, elapsed_seconds)
            for key, value in rounded_phase_totals.items()
            if key != "blocking_action"
        }
        request_profile = {
            key: (
                rounded_seconds(value)
                if isinstance(value, float) else value
            )
            for key, value in dict(overall_request_stats).items()
        }
        task_record = {
            "instruction": args.instruction,
            "model": args.model,
            "base_url": args.base_url,
            "output_dir": output_dir,
            "request_log_path": request_log_path,
            "start_time": task_start_time,
            "end_time": task_end_time,
            "elapsed_seconds": elapsed_seconds,
            "status": task_status,
            "steps_recorded": len(history),
            "log_path": task_log_path,
            "profile_mode": args.profile,
        }
        if error_message:
            task_record["error"] = error_message

        append_task_log(task_log_path, task_record)

        if profile_enabled:
            task_summary = {
                "instruction": args.instruction,
                "model": args.model,
                "base_url": args.base_url,
                "profile_mode": args.profile,
                "status": task_status,
                "start_time": task_start_time,
                "end_time": task_end_time,
                "elapsed_seconds": elapsed_seconds,
                "steps_recorded": len(history),
                "completed_steps": completed_steps,
                "paths": {
                    "task_log_path": task_log_path,
                    "request_log_path": request_log_path,
                    "step_profile_path": step_profile_path,
                    "action_profile_path": action_profile_path if detailed_profile else None,
                    "task_summary_path": task_summary_path,
                },
                "phase_totals_seconds": rounded_phase_totals,
                "phase_percentages": phase_percentages,
                "request_profile": request_profile,
                "action_totals_seconds": {
                    key: rounded_seconds(value)
                    for key, value in sorted(
                        action_totals.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                },
                "slowest_step": slowest_step,
                "slowest_action": slowest_action,
            }
            if error_message:
                task_summary["error"] = error_message
            write_json_file(task_summary_path, task_summary)

        print(f"\n[TASK END] {task_end_time}")
        print(f"[TASK STATUS] {task_status}")
        print(f"[TASK DURATION] {elapsed_seconds}s")
        print(f"[TASK LOGGED TO] {task_log_path}")
        if profile_enabled:
            print("[PROFILE BREAKDOWN]")
            for key, value in sorted(
                phase_percentages.items(),
                key=lambda item: rounded_phase_totals.get(item[0], 0.0),
                reverse=True,
            ):
                print(
                    f"  {key}: {rounded_phase_totals.get(key, 0.0)}s "
                    f"({value}%)"
                )
            if request_profile:
                print(
                    "[PROFILE REQUESTS] "
                    f"send={int(request_profile.get('send_count', 0))} "
                    f"receive={int(request_profile.get('receive_count', 0))} "
                    f"payload_prepare={request_profile.get('payload_prepare_seconds', 0.0)}s "
                    f"network_latency={request_profile.get('network_latency_seconds', 0.0)}s "
                    f"errors={int(request_profile.get('error_count', 0))}"
                )
            if slowest_step:
                print(
                    "[PROFILE SLOWEST STEP] "
                    f"step={slowest_step['step_id']} "
                    f"duration={slowest_step['duration_seconds']}s "
                    f"dominant_phase={slowest_step.get('dominant_phase')}"
                )
            if slowest_action:
                print(
                    "[PROFILE SLOWEST ACTION] "
                    f"step={slowest_action['step_id']} "
                    f"action={slowest_action['action_type']} "
                    f"duration={slowest_action['duration_seconds']}s"
                )
            print(f"[PROFILE SUMMARY WRITTEN] {task_summary_path}")


if __name__ == "__main__":
    main()
