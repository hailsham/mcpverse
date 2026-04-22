import os
import json
import time
import pathlib
import argparse
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv


from threading import Lock
import pandas as pd
import importlib
import asyncio
from dataclasses import dataclass, field

from concurrent.futures import ThreadPoolExecutor, as_completed
from openai.types.chat import ChatCompletion

from camel.toolkits import CodeExecutionToolkit
from camel.logger import get_logger, set_log_level, set_log_file
from camel.toolkits import MCPToolkit

from judger import LLMJudger
from utils import read_data, write_data
from mcp_agent_runner import MCPAgentRunner


logger = get_logger(__name__)


@dataclass
class EvaluatorConfig:
    """Evaluator configuration class to avoid hardcoding"""
    # Debug configuration
    test_id: str = 'all'
    debug_mcp: List[str] = field(default_factory=lambda: ['filesystem', 'fetch', 'time'])
    MCPVerse_ROOT = os.getcwd()
    debug_question: str = f"How many files in {MCPVerse_ROOT}/test_data/txt"
    


    
    # MCP tool list configuration
    maxscale_tools: List[str] = field(default_factory=lambda: [
        "amap-maps", "fetch", "mcp-deepwiki", "howtocook-mcp", "variflight",
        "time", "excel", "memory", "sqlite", "airbnb", "alphavantage",
        "calculator", "dataset-viewer", "mindmap", "rijksmuseum-server",
        "nasa-mcp", "datagov", "world_bank",
        "context7", "mcp-server-hotnews", "filesystem", "git", "puppeteer", "exa", 
        "appinsightmcp", 
         "Bazi", "investor", "arxiv-mcp-server",
        "kospi-kosdaq", "whois", "weibo", "anilist", "berlin-transport",
        "yahoo-finance", "wuwa-mcp",
        "bilibili", "metmuseum-mcp", "12306-mcp",
        "wikipedia", "3rd_party_mcp_server_shuidi"
    ])
    
    standard_tools: List[str] = field(default_factory=lambda: [
        'context7', 'world_bank', 'git', 'variflight',
        'excel', 'dataset-viewer', 'kospi-kosdaq', 'wikipedia',
        'nasa-mcp', 'yahoo-finance', 'amap-maps',
        'filesystem', 'whois', 'sqlite', 'Bazi',
        'wuwa-mcp', 'datagov', 'bilibili', 
        '12306-mcp', 'arxiv-mcp-server', 
        'fetch', 'investor', '3rd_party_mcp_server_shuidi', 'time'
    ])




class MCPParser:
    """MCP string parsing utility class"""
    
    # Supported separators, sorted by priority
    SEPARATORS = ['+', ';']
    
    @classmethod
    def parse(cls, mcp_string: str) -> List[str]:
        """
        Parse MCP string with support for multiple separators
        
        Args:
            mcp_string: MCP string that may contain '+' or ';' separators
            
        Returns:
            List of MCP names
            
        Examples:
            >>> MCPParser.parse("tool1+tool2+tool3")
            ['tool1', 'tool2', 'tool3']
            
            >>> MCPParser.parse("tool1;tool2")
            ['tool1', 'tool2']
            
            >>> MCPParser.parse("single_tool")
            ['single_tool']
        """
        if not mcp_string or pd.isna(mcp_string):
            return []
        
        mcp_string = str(mcp_string).strip()
        
        for separator in cls.SEPARATORS:
            if separator in mcp_string:
                return [m.strip() for m in mcp_string.split(separator) if m.strip()]
        
        return [mcp_string] if mcp_string else []


class Evaluator:
    """Refactored evaluator class"""
    
    def __init__(self, args, config: Optional[EvaluatorConfig] = None):
        self.args = args
        self.config = config or EvaluatorConfig()
        
        # Load environment variables
        base_dir = pathlib.Path(__file__).parent.parent
        env_path = base_dir / ".env"
        load_dotenv(dotenv_path=str(env_path), override=True)
        
        self._setup_paths()
        self._setup_logging()
        
        # Pass inout_folder as output_folder for MCPAgentRunner
        args.output_folder = args.inout_folder
        self.runner = MCPAgentRunner(args)
        self.judger = LLMJudger(args)
        
        # No need to load dataset in debug mode
        if 'debug' in args.mode:
            self.df = None
            self.logs = {}
        else:
            self.df, self.logs = self._load_data()
        
        self.lock = Lock()
    
    def _setup_paths(self):
        """Set up file paths"""
        self.model_name = self.args.model_name
        self.infer_mode = self.args.infer_mode
        self.fc_mode = self.args.fc_mode
        self.workers = self.args.workers
        self.tool_path = self.args.tool_path
        self.inout_folder = self.args.inout_folder
        self.dataset_path = self.args.dataset_path
        self.eval_parallel = self.args.eval_parallel
        
        # Set up paths based on inout_folder
        if 'debug' in self.args.mode:
            self.output_dir = "outputs/debug"
            self.inout_path = f"{self.output_dir}/debug.csv"
            self.log_dir = f"{self.output_dir}/logs"
            self.rollout_path = f"{self.log_dir}/debug.json"
            self.log_path = f"{self.log_dir}/debug.log"
        else:
            self.output_dir = f"outputs/{self.inout_folder}"
            self.inout_path = f"{self.output_dir}/{self.inout_folder}.csv"
            self.log_dir = f"{self.output_dir}/logs"
            self.rollout_path = f"{self.log_dir}/{self.inout_folder}.json"
            self.log_path = f"{self.log_dir}/{self.inout_folder}.log"
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/tmp", exist_ok=True)
        os.makedirs(f"{self.output_dir}/task_outputs", exist_ok=True)
        
    def _setup_logging(self):
        """Set up logging"""
        set_log_file(self.log_path)
        set_log_level(level="INFO")
        
    def _load_data(self) -> tuple:
        """Load data"""
        if os.path.isfile(self.inout_path):
            logger.info(f"Resuming data from {self.inout_path}")
            df = read_data(self.inout_path)
        else:
            dataset_path = self.dataset_path
            df = read_data(dataset_path)
        
        # Replace output subfolder placeholder with inout_folder
        df = df.applymap(lambda x: x.replace('{OUTPUT_SUB_FOLDER}', self.inout_folder) if isinstance(x, str) else x)
        
        # Load rollout logs
        logs = {}
        if os.path.isfile(self.rollout_path):
            with open(self.rollout_path, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        
        # Ensure result columns exist
        if f'{self.model_name}-answer' not in df.columns:
            df[f'{self.model_name}-answer'] = None
        if f'{self.model_name}-score' not in df.columns:
            df[f'{self.model_name}-score'] = None
            
        return df, logs
    
    def _should_process_row(self, row, check_answer: bool = True) -> bool:
        """
        Determine whether this row should be processed
        
        Args:
            row: Data row
            check_answer: Whether to check the answer column
        """
        qid = row.get('question_id', '')
        
        # Check TEST_ID filter
        if self.config.test_id != 'all' and qid != self.config.test_id:
            return False
        
        # Check if answer already exists
        if check_answer and not pd.isnull(row.get(f'{self.model_name}-answer')):
            logger.info(f"Skipping {qid}: answer already exists")
            return False
        
        return True
    
    def dump_results(self, stage: str, index: int, qid: str, results: Dict[str, Any]):
        """
        Save results to file
        
        Unified result saving logic
        """
        with self.lock:
            try:
                if stage == 'infer':
                    self._dump_infer_results(index, qid, results)
                elif stage == 'eval':
                    self._dump_eval_results(index, results)
                elif stage == 'get_ref':
                    self._dump_ref_results(index, results)
                else:
                    logger.warning(f"Unknown stage: {stage}")
            except Exception as e:
                logger.error(f"Failed to save results ({stage}, {qid}): {e}")
                raise
    
    def _dump_infer_results(self, index: int, qid: str, results: Dict[str, Any]):
        """Save inference results"""
        self.df.loc[index, f'{self.model_name}-answer'] = results.get('answer')
        
        # Serialize ChatCompletion objects in memory
        memory = results.get('memory', [])
        if isinstance(memory, list):
            for idx, item in enumerate(memory):
                if isinstance(item, ChatCompletion):
                    memory[idx] = str(item)
        
        self.logs[qid] = {
            'rollout': memory,
            'answer': results.get('answer')
        }
        
        write_data(self.inout_path, self.df)
        
        with open(self.rollout_path, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, indent=4, ensure_ascii=False)
        
        logger.info(f"=> Saved results to {self.inout_path}")
        logger.info(f"=> Saved rollout to {self.rollout_path}")
    
    def _dump_eval_results(self, index: int, results: Dict[str, Any]):
        """Save evaluation results"""
        self.df.loc[index, f'{self.model_name}-score'] = results.get('score')
        self.df.loc[index, 'judge-reason'] = results.get('reason', '')
        write_data(self.inout_path, self.df)
    
    def _dump_ref_results(self, index: int, results: Dict[str, Any]):
        """Save reference answers"""
        os.makedirs(os.path.dirname(self.inout_path), exist_ok=True)
        self.df.loc[index, 'answer'] = results.get('answer')
        write_data(self.inout_path, self.df)
    
    async def predict_row(self, index: int, row, tools) -> Optional[Dict[str, Any]]:
        """Async prediction for a single row"""
        try:
            if self.fc_mode == 'FC':
                agent = self.runner.build_agent(tools=tools)
            elif self.fc_mode == 'Prompt':
                agent = self.runner.build_mcp_prompt_agent(tools=tools)
            else:
                raise ValueError(f"Unknown fc_mode: {self.fc_mode}")
            
            task = row['question']
            logger.info(f"=> {row['question_id']}: {task}")
            
            return await self.runner.run_task(agent, task)
        except Exception as e:
            qid = row.get('question_id', 'unknown')
            error_msg = f"[ERROR] Prediction failed: {str(e)}"
            logger.error(f"Prediction failed ({qid}): {e}")
            # Return error as answer to avoid infinite retries
            return {
                'answer': error_msg,
                'memory': []
            }
    
    def _predict_with_tools(self, index: int, row, tools) -> Optional[Dict[str, Any]]:
        """Synchronous prediction with tools"""
        try:
            agent = self.runner.build_agent(tools=tools)
            task = row['question']
            logger.info(f"=> {row['question_id']}: {task}")
            return self.runner.run_task_sync(agent, task)
        except Exception as e:
            qid = row.get('question_id', 'unknown')
            error_msg = f"[ERROR] Prediction failed: {str(e)}"
            logger.error(f"Prediction failed ({qid}): {e}")
            # Return error as answer to avoid infinite retries
            return {
                'answer': error_msg,
                'memory': []
            }
    
    def predict_row_sync(self, index: int, row) -> tuple:
        """Synchronous prediction for a single row (Oracle mode)"""
        qid = row['question_id']
        task = row['question']
        logger.info(f"=> {qid}: {task}")
        
        mcp_list = MCPParser.parse(row.get('MCP', ''))
        
        try:
            tools = []
            
            # Load code execution tool
            if 'code' in mcp_list:
                logger.info("=> Loading tool: code")
                tools.extend(
                    CodeExecutionToolkit(sandbox="subprocess", verbose=False).get_tools()
                )
            
            # Filter out non-code MCPs
            non_code_mcps = [m for m in mcp_list if m != 'code']
            
            if not non_code_mcps:
                return index, self._predict_with_tools(index, row, tools)
            
            # Load MCP tools
            config = self.runner._load_mcp_config(non_code_mcps)
            with MCPToolkit(config_dict=config, timeout=40) as mcp_toolkit:
                tools.extend(mcp_toolkit.get_tools())
                return index, self._predict_with_tools(index, row, tools)
                
        except Exception as e:
            error_msg = f"[ERROR] Prediction failed: {str(e)}"
            logger.error(f"predict_row_sync failed ({qid}): {e}")
            # Return error as answer to avoid infinite retries
            return index, {
                'answer': error_msg,
                'memory': []
            }
    
    def evaluate_row(self, index: int, row) -> Optional[Dict[str, Any]]:
        """Evaluate a single row"""
        eval_method = row.get('eval_method', 'llm_as_a_judge')
        pred = row.get(f'{self.model_name}-answer')
        answer = row.get('answer')
        qid = row.get('question_id')
        question = row.get('question')
        
        # Initialize return values
        score = None
        reason = ""
        
        try:
            if eval_method == 'llm_as_a_judge':
                score, reason = self.judger.judge(question, pred, answer)
                
            elif eval_method == 'eval_script':
                try:
                    logger.info(f"=> Importing eval_scripts.{qid}")
                    module = importlib.import_module(f"eval_scripts.{qid}")
                    passed = module.run_test(pred, answer)
                    score = 1 if passed else 0
                except ModuleNotFoundError:
                    raise KeyError(f"Test module '{qid}' not found")
                except Exception as e:
                    logger.error(f"[TestCaseError] {e}")
                    score = 0
            else:
                logger.warning(f"Unknown evaluation method: {eval_method}")
            logger.warning(f"score: {score}")
            return {'score': score, 'reason': reason}
            
        except Exception as e:
            logger.error(f"Evaluation failed ({qid}): {e}")
            return None
    
    def get_reference_row(self, ref_id: str, max_retries: int = 3) -> Dict[str, Any]:
        """Get reference answer"""
        for retry in range(max_retries):
            try:
                module = importlib.import_module(f"templates.{ref_id}")
                answer = module.get_reference()
                return {'answer': answer}
            except Exception as e:
                logger.warning(
                    f"Failed to get reference answer {ref_id} (attempt {retry + 1}/{max_retries}): {e}"
                )
                if retry < max_retries - 1:
                    time.sleep(10)
        
        return {'answer': None}
    
    def get_references(self):
        """Get all reference answers"""
        logger.info("=> Starting getting reference answer")
        for i, row in self.df.iterrows():
            qid = row['question_id']
            ref_id = row.get('get_answer_id')

            if not ref_id:
                continue

            # time-sensitive questions use tq* templates
            if row.get('time-sensitive') == 'Yes':
                ref_id = f"t{ref_id}"
            elif row.get('time-sensitive') != 'No':
                continue
            
            logger.info(f"=> Getting reference answer: {qid}")
            
            try:
                results = self.get_reference_row(ref_id)
                if results.get('answer') is not None:
                    logger.info(f"===> Done: {qid}")
                    self.dump_results('get_ref', i, qid, results)
            except Exception as e:
                logger.error(f"===> Failed!! {qid}: {ref_id}/{e}")

        # Filter out rows that failed to get reference answers
        before = len(self.df)
        self.df = self.df[self.df['answer'].notna()].reset_index(drop=True)
        after = len(self.df)
        if before != after:
            logger.info(f"=> Filtered out {before - after} rows without reference answers, keeping {after}")
            write_data(self.inout_path, self.df)
        logger.info("=>[Done] Getting reference answer")
    
    def get_target_tools(self) -> List[str]:
        """Get target tool list"""
        if self.infer_mode == 'maxscale':
            tool_list = self.config.maxscale_tools
        elif self.infer_mode == 'standard':
            tool_list = self.config.standard_tools
        else:
            return []
        
        # Deduplicate and expand
        mcp_list = []
        for item in tool_list:
            parts = str(item).split('+')
            mcp_list.extend([p.strip() for p in parts if p.strip()])
        
        return list(set(mcp_list))
    
    async def run_with_all_mcp(self):
        """Run with all MCPs"""
        mcp_list = self.get_target_tools()
        tools = await self.runner.connect(mcp_list)
        
        failed_clients = self.runner.get_failed_tools()
        logger.info(f"=> Failed clients: {failed_clients}")
        logger.info(f"=> Total tools: {len(tools)}")
        
        # Retry failed connections
        retry_count = 0
        max_retries = 3
        while failed_clients and retry_count < max_retries:
            logger.info(f"=> Retrying failed clients (attempt {retry_count + 1})")
            failed_tools = await self.runner.connect(failed_clients)
            tools.extend(failed_tools)
            failed_clients = self.runner.get_failed_tools()
            retry_count += 1
        
        # Add code execution tool
        logger.info("=> Adding tool: code execution")
        tools.extend(
            CodeExecutionToolkit(sandbox="subprocess", verbose=False).get_tools()
        )
        
        logger.info(f"=> Final tool count: {len(tools)}")
        logger.info(f"=> Final failed clients: {failed_clients}")
        
        try:
            for index, row in self.df.iterrows():
                if row.get('eval_method') == 'test_case':
                    continue
                
                if not pd.isnull(row.get(f'{self.model_name}-answer')):
                    logger.info(f"Skipping {row['question_id']}")
                    continue
                
                results = await self.predict_row(index, row, tools)
                if results is not None:
                    self.dump_results('infer', index, row['question_id'], results)
        finally:
            await self.runner.disconnect()
    
    def run_with_oracle_mode_sync(self):
        """Synchronous run in Oracle mode"""
        for index, row in self.df.iterrows():
            qid = row['question_id']
            logger.info(f"=> Processing: {qid}")
            
            if not self._should_process_row(row):
                continue
            
            try:
                _, results = self.predict_row_sync(index, row)
                if results is not None:
                    self.dump_results('infer', index, qid, results)
            except Exception as e:
                logger.error(f"=> Error ({qid}): {e}")
    
    async def debug(self):
        """Debug mode"""
        mcp_list_debug = self.config.debug_mcp
        
        tools = await self.runner.connect(mcp_list_debug)
        tools.extend(
            CodeExecutionToolkit(sandbox="subprocess", verbose=False).get_tools()
        )
        
        row = {
            'question_id': "debug-000",
            'question': self.config.debug_question
        }
        
        try:
            results = await self.predict_row(0, row, tools)
        finally:
            await self.runner.disconnect()
    
    def _run_eval_sequential(self):
        """Sequential evaluation"""
        for index, row in self.df.iterrows():
            qid = row['question_id']
            logger.info(f"=> Processing: {qid}")
            
            # Check TEST_ID filter
            if self.config.test_id != 'all' and qid != self.config.test_id:
                continue
            
            # Check if score already exists
            if not pd.isnull(row.get(f'{self.model_name}-score')):
                logger.info(f"Skipping {qid}")
                continue
            
            results = self.evaluate_row(index, row)
            if results is not None:
                self.dump_results('eval', index, qid, results)
    
    def _run_eval_parallel(self):
        """Parallel evaluation"""
        futures = {}
        
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            for index, row in self.df.iterrows():
                future = executor.submit(self.evaluate_row, index, row)
                futures[future] = index
            
            for future in as_completed(futures):
                index = futures[future]
                try:
                    results = future.result()
                    if results is not None:
                        qid = self.df.loc[index, 'question_id']
                        self.dump_results('eval', index, qid, results)
                except Exception as e:
                    logger.error(f"=> Unhandled error in worker thread: {e}")
    
    
    def calculate_stats(self):
        """Calculate stats"""
        df = self.df
        score_col = f'{self.model_name}-score'
        comp_col = 'complexity'
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
        overall_mean = df[score_col].mean()
        by_complexity = df.groupby(comp_col, dropna=False)[score_col].mean().reset_index().rename(columns={score_col: "avg_score"}).sort_values("avg_score", ascending=False)
        print(f"{self.model_name}:{self.infer_mode}: {overall_mean:.4f}")
        print(by_complexity.to_string(index=False))



    def eval(self):
        """Execute evaluation"""
        if self.eval_parallel:
            self._run_eval_parallel()
        else:
            self._run_eval_sequential()

        # Calculate stats
        self.calculate_stats()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run selected MCP tools.")
    
    parser.add_argument(
        '--mode',
        help='Run mode: debug, infer, eval, get_ref, all',
        nargs='+',
        default=['all'],
        type=str,
        choices=['debug', 'get_ref', 'infer', 'eval', 'all']
    )
    parser.add_argument("--model_name", type=str, default="deepseek-v3")
    parser.add_argument("--tool_path", type=str, default="tool_full.json")
    parser.add_argument("--inout_folder", type=str, default="", help="Output folder name under outputs/")
    parser.add_argument(
        "--infer_mode",
        type=str,
        default="oracle",
        help="Inference mode: oracle, standard, maxscale"
    )
    parser.add_argument("--fc_mode", type=str, default="FC")
    parser.add_argument("--judge_model", type=str, default="QwQ")
    parser.add_argument("--eval_parallel", action="store_true", help="Run evaluation in parallel")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--dataset_path", type=str, default="")
    
    # New: Support command line override for debug configuration
    parser.add_argument("--test_id", type=str, default="all", help="Specify question ID to test")
    parser.add_argument("--debug_mcp", nargs='+', default=None, help="MCP list for debug mode")
    parser.add_argument("--debug_question", type=str, default=None, help="Question for debug mode")
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Validate arguments: non-debug mode requires inout_folder
    if 'debug' not in args.mode and not args.inout_folder:
        raise ValueError("--inout_folder argument is required for non-debug mode")
    
    # Create configuration with command line override support
    config = EvaluatorConfig(
        test_id=args.test_id,
        debug_mcp=args.debug_mcp or EvaluatorConfig().debug_mcp,
        debug_question=args.debug_question or EvaluatorConfig().debug_question
    )
    
    evaluator = Evaluator(args, config)
    
    # Debug mode
    if 'debug' in args.mode:
        asyncio.run(evaluator.debug())
    
    # Get reference answers
    if any(mode in ['all', 'get_ref'] for mode in args.mode):
        evaluator.get_references()
    
    # Inference stage
    if any(mode in ['all', 'infer'] for mode in args.mode):
        if args.infer_mode == 'oracle':
            evaluator.run_with_oracle_mode_sync()
        elif args.infer_mode in ['standard', 'maxscale']:
            asyncio.run(evaluator.run_with_all_mcp())
    
    # Evaluation stage
    if any(mode in ['all', 'eval'] for mode in args.mode):
        evaluator.eval()


if __name__ == "__main__":
    main()

