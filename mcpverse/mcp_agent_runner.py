import json
import os
import sys
import time
import re

import argparse
import importlib
import asyncio

from datetime import date
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from camel.agents import ChatAgent, MCPPromptAgent
from camel.configs import ChatGPTConfig
from camel.logger import set_log_level, get_logger, set_log_file
from camel.models import ModelFactory
from camel.toolkits import MCPToolkit
from camel.types import ModelPlatformType, ModelType

from utils import read_data, write_data


logger = get_logger(__name__)


class MCPAgentRunner:
    def __init__(self, args):
        self.args = args
        self.model_name = args.model_name
        self.tool_path = args.tool_path
        # self.selected_tools = args.tools or "all"
        self.output_folder = args.output_folder

        self.mcp_toolkit = None
        self.model = None
        self.agent = None
        self.tools = []


    def _load_mcp_config(self, mcp_list: List) -> dict:
        config_path = Path(__file__).parent / self.tool_path

        os.environ["MCPVERSE_ROOT"] = str(Path(__file__).parent.parent)
        print(os.environ["MCPVERSE_ROOT"])

        with open(config_path, "r", encoding="utf-8") as f:
            raw = f.read()

        raw = raw.replace("${MCPVERSE_ROOT}", os.environ["MCPVERSE_ROOT"])
        raw = raw.replace("${SMITHERY_API_KEY}", os.environ["SMITHERY_API_KEY"])
        raw = raw.replace("${AMAP_API_KEY}", os.environ["AMAP_API_KEY"])
        raw = raw.replace("${ALPHAVANTAGE_API_KEY}", os.environ["ALPHAVANTAGE_API_KEY"])
        raw = raw.replace("${RIJKSMUSEUM_API_KEY}", os.environ["RIJKSMUSEUM_API_KEY"])
        raw = raw.replace("${NASA_API_KEY}", os.environ["NASA_API_KEY"])
        raw = raw.replace("${VARFLIGHT_API_KEY}", os.environ["VARFLIGHT_API_KEY"])


        expanded = os.path.expandvars(raw)
        full_config = json.loads(expanded)

        if mcp_list == ["none"]:
            return {"mcpServers": {}}
        if mcp_list == "T128":
            selected = {
                tool: full_config["mcpServers"][tool]
                for tool in mcp_list
                if tool in full_config.get("mcpServers", {})
            }
            return {"mcpServers": selected}
        elif mcp_list == ["all"]:
            return full_config
        else:
            logger.info(f"=> loading target tools: {mcp_list}")
            selected = {
                tool: full_config["mcpServers"][tool]
                for tool in mcp_list
                if tool in full_config.get("mcpServers", {})
            }
            return {"mcpServers": selected}

    def _init_model(self):
        logger.info(f"=> load model: {self.model_name}")
        if self.model_name == 'deepseek-v3':
            self.model = ModelFactory.create(
                model_platform=ModelPlatformType.DEEPSEEK,
                model_type=ModelType.DEEPSEEK_CHAT,
                model_config_dict=dict(temperature=0.1, max_tokens=60000),
            )
        elif self.model_name == 'deepseek-v3.2':
            self.model = ModelFactory.create(
                model_platform=ModelPlatformType.DEEPSEEK,
                model_type=ModelType.DEEPSEEK_CHAT,
                model_config_dict=dict(temperature=0.1, max_tokens=60000),
            )
        elif self.model_name == 'deepseek-r1':
            self.model = ModelFactory.create(
                model_platform=ModelPlatformType.DEEPSEEK,
                model_type=ModelType.DEEPSEEK_REASONER,
                model_config_dict=dict(temperature=0.1, max_tokens=60000),
            )
        elif self.model_name == 'claude-3-7-sonnet-20250219':
            self.model = ModelFactory.create(
                model_platform=ModelPlatformType.ANTHROPIC,
                model_type='claude-3-7-sonnet-20250219',
                model_config_dict=dict(
                    max_tokens=64000,                
                )
            )
        elif self.model_name == 'claude-sonnet-4-20250514':
            self.model = ModelFactory.create(
                model_platform=ModelPlatformType.ANTHROPIC,
                model_type='claude-sonnet-4-20250514',
                model_config_dict=dict(
                    max_tokens=32000,                
                )
            )
        elif self.model_name == 'claude-sonnet-4-20250514-ws':
            self.model = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
                model_type='MaaS_Sonnet_4',
                url='https://genaiapi.cloudsway.net/v1/ai/ugLMeREFyehDfnrC',
                api_key=os.getenv("ANTHROPIC_WS_API_KEY"),
                model_config_dict=dict(
                    max_tokens=60000,                
                )
            )
        elif self.model_name == 'gpt4o':
            self.model = ModelFactory.create(
                model_platform=ModelPlatformType.AZURE,
                model_type=os.getenv("AZURE_OPENAI_MODEL_TYPE"),
                model_config_dict=dict(temperature=0.1, max_tokens=100000),
            )
        elif self.model_name == 'gemini25-pro':
            from google.auth import default
            from google.auth.transport.requests import Request

            credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            credentials.refresh(Request())

            self.model = ModelFactory.create(
                model_platform=ModelPlatformType.GEMINI,
                api_key=credentials.token,
                model_type='google/gemini-2.5-pro',
                model_config_dict=dict(temperature=0.1, max_tokens=60000),
            )
        elif self.model_name == 'Qwen3-235B-non-thinking':
            self.model = ModelFactory.create(
                model_platform=ModelPlatformType.QWEN,
                model_type='qwen3-235b-a22b',
                model_config_dict=dict(
                    max_tokens=110000,
                    temperature=0.7,
                    top_p=0.8,
                    stream=False,
                    extra_body={"enable_thinking": False},
                ),
            )
        elif self.model_name == '30a3b':
            self.model = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
                model_type='30a3b',
                url='http://10.210.6.10:23232/v1',
                api_key='xx',
                model_config_dict=dict(temperature=0.0, max_tokens=32000),
            )
        elif self.model_name == 'Qwen3-30B-A3B':
            self.model = ModelFactory.create(
                model_platform=ModelPlatformType.QWEN,
                model_type='qwen3-30b-a3b',
                model_config_dict=dict(
                    max_tokens=110000,
                    temperature=0.7,
                    top_p=0.8,
                    stream=False,
                    extra_body={"enable_thinking": False},
                ),
            )
        elif self.model_name == 'Qwen3-32B':
            self.model = ModelFactory.create(
                model_platform=ModelPlatformType.QWEN,
                model_type='qwen3-32b',
                model_config_dict=dict(
                    max_tokens=110000,
                    temperature=0.7,
                    top_p=0.8,
                    stream=False,
                    extra_body={"enable_thinking": False},
                ),
            )
        elif self.model_name == 'gpt-5':
            self.model = ModelFactory.create(
                model_platform=ModelPlatformType.AZURE,
                model_type='gpt-5',
                model_config_dict=dict(max_completion_tokens=8 * 1024),
            )
        elif self.model_name == 'Qwen2-5-72B-Instruct':
            self.model = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
                model_type='Qwen2.5-72B-Instruct',
                url='http://10.210.1.23:5162/v1',
                api_key='EMPTY',
                model_config_dict=dict(temperature=0.0, max_tokens=32000),
            )
        else:
            raise ValueError(f"Model {self.model_name} not supported")
    
    def _ensure_model(self):
        """Ensure model is initialized (lazy loading)"""
        if self.model is None:
            self._init_model()

    def build_agent(self, tools):
        self._ensure_model()
        sys_msg = f"""
You are a helpful assistant. your goal is to complete a task. Please note that the task may be very complicated. Do not attempt to solve the task by single step. Here are some tips may help you:
<tips>
- your various mcp tools to use, such as search toolkit, map toolkit, document relevant toolkit, code execution toolkit, etc.
- If one way fails to provide an answer, try other ways or methods. The answer does exists.
- When looking for specific numerical values (e.g., dollar amounts), prioritize reliable sources and avoid relying only on search snippets.  
- you can write python code to solve the task if needed.
- If you want to generate temporary files or download temporary files, please place them in the `./outputs/{self.output_folder}/tmp/` folder. 
- Some tools need internet access. If you run into connection issues while using them, try retrying the request.
</tips>
""".strip()
        agent = ChatAgent(system_message=sys_msg, tools=tools, model=self.model)
        return agent

        
    def build_mcp_prompt_agent(self, tools):
        tools_info_list = []
        for tool in tools:

            fc= tool.openai_tool_schema['function']
            tools_info_list.append(fc)
        sys_msg = f'''
You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.  Please note that the task may be very complicated. Do not attempt to solve the task by single step. 

You should **only** return the function calls in your response until you know the exact answer of the problem. For each time you call a function, user will return the corresponding execution result to you with the format `<tool_response>{{result}}</tool_response>`. You can use the result to make further function calls.

<tips>
1. Put **all** calls in **a single list of JSON objects**, with the format:  
   `[{{"name": "function_name_1", "parameters": {{ "param_1A": value1A, "param_1B": value1B }}}}, {{"name": "function_name_2", "parameters": {{ "param_2A": value2A, "param_2B": value2B }}}}, ...]`  
   Never output multiple separate lists or include any text before or after the list. 
2. Do **not** include any other text alongside the list or use other format. Do not mock any responses of the functions, just only return a function call.
3. Your task may span multiple rounds of function calls. If any function depends on the results of previous calls, you must first obtain the prerequisite result before invoking the dependent function.
4. If one way fails to provide an answer, try other ways or methods. The answer does exists.
5. When looking for specific numerical values (e.g., dollar amounts), prioritize reliable sources and avoid relying only on search snippets.  
6. You can write python code to solve the task if needed.
7. If you want to generate any files, please place them in the `outputs` folder. 
8. Some tools need internet access. If you run into connection issues while using them, try retrying the request.
9. If the problem is fully solved and no further function calls are needed, return a **concise text summary of the final answer** instead of a function list. Otherwise, always return a function list.
</tips>

You are provided with function signatures within <tools></tools> XML tags:
<tools> 
{tools_info_list}
</tools> 
'''.strip()
        
        sys_msg2 = """
You are a helpful assistant. your goal is to complete a task. Please note that the task may be very complicated. Do not attempt to solve the task by single step. Here are some tips may help you:
<tips>
- your various mcp tools to use, such as search toolkit, map toolkit, document relevant toolkit, code execution toolkit, etc.
- If one way fails to provide an answer, try other ways or methods. The answer does exists.
- When looking for specific numerical values (e.g., dollar amounts), prioritize reliable sources and avoid relying only on search snippets.  
- you can write python code to solve the task if needed.
- If you want to generate any files, please place them in the `outputs` folder. 
- Some tools need internet access. If you run into connection issues while using them, try retrying the request.
</tips>

You are provided with function signatures within <tools></tools> XML tags:
<tools> 
{tools_info_list}
</tools> 


<tool_call_tips>
- Put **all** calls in **a single list of JSON objects**, with the format:  
`[{{"name": "function_name_1", "parameters": {{ "param_1A": value1A, "param_1B": value1B }}}}]` 
- IMPORTANT!!!: Unless you are confident that you can produce the **final answer**, you should **always output a tool_call** with above format instead.
</tool_call_tips>

"""

        self._ensure_model()
        agent = MCPPromptAgent(
            system_message=sys_msg2, 
            model=self.model,
            tools=tools, 
        )
        return agent

    def get_tools(self, mcp_name):
        if mcp_name == "all":
            return [item for sublist in self.tool_dict.values() for item in sublist]
        elif mcp_name:
            return self.tool_dict[mcp_name]

    def get_failed_tools(self):
        return self.mcp_toolkit.failed_clients

    async def connect(self, mcp_list: List):
        logger.info("=> connect")
        config = self._load_mcp_config(mcp_list)
        self.mcp_toolkit = MCPToolkit(config_dict=config, timeout=20)

        t1 = time.time()
        await self.mcp_toolkit.connect()
        tools = self.mcp_toolkit.get_tools()

        t2 = time.time()
        logger.info(f"=> mcp init done, time: {t2 - t1}")
        return tools

    def tool_test(self):
        logger.info("=> connect")
        config = self._load_mcp_config()
        with MCPToolkit(config_dict=config, timeout=30) as toolkit:
            tools = toolkit.get_tools()
            for tool in tools:
                print(tool.get_function_name())


    async def disconnect(self):
        logger.info("=> Shutting down...")
        await asyncio.sleep(1)
        if self.mcp_toolkit:
            await self.mcp_toolkit.disconnect()

    async def run_task(self, agent, task):
        result = await agent.astep(task)
        # Handle case where astep() returns (ChatAgentResponse, trace_id) tuple
        response = result[0] if isinstance(result, tuple) else result
        if response.terminated:
            output = {
                'answer': response.info['termination_reasons'][0],
                'memory': ""
            }
        else:
            output = {
                'answer': response.msgs[0].content, 
                'memory': agent.memory.get_context()[0]
            }

        print(f"\033[94mAnswer: {output['answer']}\033[0m")
        logger.info(f"=> Answer: {output['answer']}")

        return output

    def run_task_sync(self, agent, task):
        result = agent.step(task)
        # Handle case where step() returns (ChatAgentResponse, trace_id) tuple
        response = result[0] if isinstance(result, tuple) else result
        if response.terminated:
            reason = response.info['termination_reasons'][0]
            logger.info(f"=> Terminated: {reason}")
            # Use retrieve() to get raw records, then truncate to avoid token limit error
            try:
                records = agent.memory.retrieve()
                # Convert records to OpenAI message format and truncate to last 20
                memory = [
                    record.memory_record.to_openai_message() 
                    for record in records
                ]
            except Exception as e:
                logger.warning(f"Failed to retrieve memory: {e}")
                memory = []
            return {
                'answer': reason,
                'memory': memory
            }
        answer = response.msgs[0].content
        memory = agent.memory.get_context()[0]
        logger.info(f"=> Answer: {answer}")

        return {
            'answer': answer, 
            'memory': memory
        }



    async def debug(self):
        await self.connect()
        tools = [*self.mcp_toolkit.get_tools()]
        logger.info(f"=> total tools:  {len(tools)}")
        agent = self.build_mcp_runner(tools)
        task = "What is the current rating of the app 'Candy Crush Saga' on the App Store and Google Play?"
        await self.run_task(agent, task)
        await self.disconnect()
        
    def debug_sync(self, task):
        self._ensure_model()
        with self.mcp_toolkit:
            tools=[*self.mcp_toolkit.get_tools()]
            agent = self.build_mcp_runner(tools, self.model)
            # task = "小明和小方是什么关系？"
            self.run_task(agent, task)


def test_eval(runner):
    score = runner.score(qid='Q101', pred="", answer="", eval_method="test_case")
    print(f"[TestCaseScore] {score}")

    ref = runner.get_reference(qid='q2')
    print(ref)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run selected MCP tools.")
    parser.add_argument(
        "--tools", 
        nargs="+", 
        help="List of tool names to run (e.g. fetch calculator excel)"
    )
    parser.add_argument("--model_name", type=str, default="deepseek-v3")
    parser.add_argument("--tool_path", type=str, default="tool.json")
    parser.add_argument("--dataset", type=str, default="1step")
    parser.add_argument("--question", type=str, default="")
    parser.add_argument("--tool_test", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--eval", action="store_true")


    args = parser.parse_args()
    logger.info(args)
    runner = MCPAgentRunner(args)
    if args.tool_test:
        # asyncio.run(runner.tool_test())
        runner.tool_test()
    elif args.debug:
        asyncio.run(runner.debug())
        # runner.debug_sync(args.question)
    elif args.eval:
        test_eval(runner)
    else:
        runner.run_sequential()
