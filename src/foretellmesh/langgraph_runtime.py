"""LangGraph scheduling over the same bounded, validated Agent execution steps.

One node per role, plus an optional deterministic Quant node. The graph holds
per-run structured state, never model weights or outcome labels. Compilation
uses no checkpointer or cache; persistence/resume is a separate future contract.
"""
from copy import deepcopy
from typing import TypedDict

from .agent_runtime import AgentRunner, ExecutionState
from .capabilities import route_plan
from .schema import ForecastInput


class GraphInput(TypedDict):
    context: ForecastInput


class GraphOutput(TypedDict):
    result: dict


class GraphState(GraphInput, GraphOutput):
    execution: ExecutionState


class LangGraphRunner(AgentRunner):
    """Drop-in run API; compile() also exposes LangGraph's local update stream.

    Model calls, including bounded schema repairs, remain inside role nodes.
    Graph edges only schedule the next stage or finalize a failure. No graph
    retry policy wraps a node, so model calls cannot silently multiply.
    """

    def compile(self, *, workflow: str = 'research_forecast', mode: str = 'base',
                capability_scope: set[str] | None = None):
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:
            raise RuntimeError('LangGraph is optional; install foretellmesh[graph] in the active environment') from exc
        # Freeze config and scope for this compiled graph. Backend/model ownership
        # stays with the caller's existing shared executor.
        executor = AgentRunner(self.config, self.backend, output_protocol=self.output_protocol,
                               response_transport=self.response_transport)
        scope = deepcopy(capability_scope)
        plan = route_plan(executor.config, workflow, mode, capability_scope=scope)
        graph = StateGraph(GraphState, input_schema=GraphInput, output_schema=GraphOutput)

        def prepare(state: GraphInput):
            return {'execution': executor._initialize(state['context'], workflow, mode, scope)}

        def quant_tools(state: GraphState):
            execution = deepcopy(state['execution'])
            executor._execute_quant(execution)
            return {'execution': execution}

        def role_node(step):
            def execute(state: GraphState):
                execution = deepcopy(state['execution'])
                executor._execute_step(execution, step)
                return {'execution': execution}
            return execute

        def finalize(state: GraphState):
            return {'result': executor._finish(state['execution'])}

        def after_node(next_name):
            def route(state: GraphState):
                return 'finalize' if state['execution']['result'] is not None else next_name
            return route

        graph.add_node('prepare', prepare)
        names = ['prepare']
        if plan.get('deterministic_steps'):
            graph.add_node('quant_tools', quant_tools); names.append('quant_tools')
        for step in plan['steps']:
            graph.add_node(step['agent'], role_node(deepcopy(step))); names.append(step['agent'])
        graph.add_node('finalize', finalize); names.append('finalize')
        graph.add_edge(START, 'prepare')
        for current, next_name in zip(names, names[1:]):
            graph.add_conditional_edges(current, after_node(next_name), list(dict.fromkeys([next_name, 'finalize'])))
        graph.add_edge('finalize', END)
        return graph.compile(name='foretellmesh_' + workflow)

    def run(self, context: ForecastInput, *, workflow: str = 'research_forecast', mode: str = 'base',
            capability_scope: set[str] | None = None) -> dict:
        graph = self.compile(workflow=workflow, mode=mode, capability_scope=capability_scope)
        # All supported paths are acyclic. This independent graph-level bound is
        # above the longest current path; role call budgets remain authoritative.
        output = graph.invoke({'context': context}, config={'max_concurrency': 1, 'recursion_limit': 16})
        return output['result']
