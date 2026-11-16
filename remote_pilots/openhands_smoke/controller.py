"""Executed only inside the pinned OpenHands 0.59 container."""
import asyncio
import json
import os
from pathlib import Path

from openhands.core.config import OpenHandsConfig
from openhands.core.config.agent_config import AgentConfig
from openhands.core.config.llm_config import LLMConfig
from openhands.core.config.sandbox_config import SandboxConfig
from openhands.core.main import run_controller
from openhands.events.action import MessageAction


async def main():
    config = OpenHandsConfig(
        default_agent='CodeActAgent', runtime='docker', max_iterations=20,
        file_store='local', file_store_path='/state/store',
        save_trajectory_path='/state/trajectory.json', enable_browser=False,
        sandbox=SandboxConfig(
            runtime_container_image=os.environ['PILOT_RUNTIME'],
            volumes=os.environ['PILOT_WORKSPACE'] + ':/workspace:rw',
            user_id=int(os.environ['PILOT_UID']), use_host_network=True,
            runtime_binding_address='127.0.0.1',
            keep_runtime_alive=False, rm_all_containers=False,
        ),
    )
    config.set_llm_config(LLMConfig(
        model='openai/qwen-local', api_key='local-probe',
        base_url='http://127.0.0.1:30001/v1', native_tool_calling=True,
        max_input_tokens=16384, max_output_tokens=1024, temperature=0,
        num_retries=1, timeout=180, disable_vision=True,
        disable_stop_word=True, caching_prompt=False,
        log_completions=True, log_completions_folder='/state/completions',
    ))
    config.set_agent_config(AgentConfig(
        enable_browsing=False, enable_jupyter=False, enable_mcp=False,
        enable_prompt_extensions=False, enable_plan_mode=False,
    ))
    state = await run_controller(
        config, MessageAction(content=Path('/state/task.txt').read_text()),
        exit_on_message=True, headless_mode=True,
    )
    status = getattr(state, 'agent_state', None)
    status = getattr(status, 'value', str(status))
    Path('/state/controller_result.json').write_text(json.dumps({
        'agent_state': status, 'last_error': str(getattr(state, 'last_error', '')),
    }, indent=2))
    if status != 'finished':
        raise SystemExit(f'Agent did not finish: {status}; inspect controller.log')


if __name__ == '__main__':
    asyncio.run(main())
