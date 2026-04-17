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


def main():
    args = parse_args()

    # Initialize tools
    computer_tools = ComputerTools()
    computer_tools.reset()
    vllm = GUIOwlWrapper(args.api_key, args.base_url, args.model)

    # Prepare output directory
    output_dir = get_output_dir()
    safe_instruction = sanitize_filename(args.instruction)
    task_log_path = os.path.join(output_dir, "task_runs.jsonl")

    history = []
    stop_flag = False
    task_status = "running"
    error_message = None
    task_start_time = current_timestamp()
    task_start_monotonic = time.monotonic()

    print(f"[TASK START] {task_start_time}")
    print(f"[TASK LOG] {task_log_path}")

    try:
        for step_id in range(args.max_steps):
            if stop_flag:
                break

            print(f"\nSTEP {step_id}:\n{'=' * 50}")

            # Capture screenshot
            screen_shot = os.path.join(output_dir, f"{safe_instruction}_{step_id}.png")
            if not computer_tools.get_screenshot(screen_shot):
                print(f"[ERROR] Failed to capture screenshot at step {step_id}")
                continue

            # Build messages and call the VLM
            messages = build_messages(
                screen_shot, args.instruction, history, args.model
            )

            output_text, _, raw_response = vllm.predict_mm(messages)

            # Prepend reasoning content if present
            thought = extract_reasoning_content(raw_response)
            if thought:
                output_text = f"<thinking>\n{thought}\n</thinking>{output_text}"

            print(output_text)

            # Extract and execute tool calls
            action_list = extract_tool_calls(output_text)

            dummy_image = Image.open(screen_shot)
            resized_height, resized_width = smart_resize(
                dummy_image.height,
                dummy_image.width,
                factor=16,
                min_pixels=3136,
                max_pixels=1003520 * 200,
            )

            for action_id, action in enumerate(action_list):
                action_parameter = action["arguments"]

                # Rescale normalized coordinates to actual pixels
                rescale_coordinates(action_parameter, resized_width, resized_height)

                # Execute the action
                should_stop = execute_action(computer_tools, action_parameter)

                if should_stop:
                    stop_flag = True
                    task_status = "completed"
                    break

                # Annotate screenshot for debugging / visualization
                annotate_screenshot(
                    screen_shot,
                    action_parameter,
                    os.path.join(
                        output_dir,
                        f"anno_{safe_instruction}_{step_id}_{action_id}.png",
                    ),
                )

            # Record history
            history.append({"output": output_text, "image": screen_shot})
            time.sleep(2)

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
        task_record = {
            "instruction": args.instruction,
            "model": args.model,
            "base_url": args.base_url,
            "output_dir": output_dir,
            "start_time": task_start_time,
            "end_time": task_end_time,
            "elapsed_seconds": elapsed_seconds,
            "status": task_status,
            "steps_recorded": len(history),
            "log_path": task_log_path,
        }
        if error_message:
            task_record["error"] = error_message

        append_task_log(task_log_path, task_record)
        print(f"\n[TASK END] {task_end_time}")
        print(f"[TASK STATUS] {task_status}")
        print(f"[TASK DURATION] {elapsed_seconds}s")
        print(f"[TASK LOGGED TO] {task_log_path}")


if __name__ == "__main__":
    main()
