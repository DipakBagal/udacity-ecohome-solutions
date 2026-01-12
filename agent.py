"""
EcoHome Energy Advisor Agent
LangGraph-based agent for smart home energy optimization
"""

from typing import TypedDict, Annotated, Literal
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from tools import ECOHOME_TOOLS


# System prompt for EcoHome agent
ECOHOME_SYSTEM_PROMPT = """You are EcoHome Energy Advisor, an expert AI assistant specializing in smart home energy optimization.

Your expertise includes:
- Solar energy systems and battery storage optimization
- HVAC efficiency and thermostat programming
- Time-of-use electricity rates and cost optimization
- Electric vehicle charging strategies
- Smart home automation for energy savings
- Renewable energy integration
- Seasonal energy management
- Energy usage analysis and recommendations

You have access to the following tools:
1. get_weather_forecast: Get weather forecasts to plan energy usage and solar generation
2. get_electricity_prices: Check current and upcoming electricity rates for cost optimization
3. query_energy_usage: Analyze historical energy usage patterns from the database
4. query_solar_generation: Review solar generation and self-consumption data
5. search_energy_tips: Search the knowledge base for specific energy-saving recommendations

Guidelines for providing assistance:
- Always greet users warmly and ask how you can help with their energy needs
- Use tools to gather data before making recommendations
- Provide specific, actionable advice based on real data when available
- Consider weather, electricity rates, and usage patterns in your recommendations
- Explain the reasoning behind your suggestions
- Quantify potential savings when possible (e.g., "This could save $50/month")
- Prioritize both cost savings and environmental impact
- Be encouraging and positive about energy efficiency efforts
- If you don't have specific data, use the search_energy_tips tool to find relevant information
- Always cite sources when using information from the knowledge base

Communication style:
- Professional but friendly and approachable
- Use clear, jargon-free language (explain technical terms when necessary)
- Structure responses with clear sections: Analysis, Recommendations, Next Steps
- Use bullet points for easy readability
- Include specific numbers and metrics when available

When users ask about:
- Current conditions: Use weather and pricing tools
- Historical usage: Use energy usage and solar generation queries
- General tips: Use search_energy_tips with relevant keywords
- Optimization: Combine multiple tools for comprehensive analysis
- Specific devices: Search knowledge base for device-specific guidance

Remember: Your goal is to help homeowners reduce energy costs, increase renewable energy usage, and minimize environmental impact through smart, data-driven decisions."""


# Agent state definition
class AgentState(TypedDict):
    """State for EcoHome agent"""
    messages: Annotated[list[BaseMessage], operator.add]
    next_step: str


class EcoHomeAgent:
    """EcoHome Energy Advisor Agent using LangGraph"""
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.7):
        """
        Initialize the EcoHome agent.
        
        Args:
            model_name: OpenAI model to use
            temperature: Temperature for LLM responses (0-1)
        """
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.llm_with_tools = self.llm.bind_tools(ECOHOME_TOOLS)
        
        # Build the graph
        self.workflow = self._build_graph()
        self.checkpointer = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self._tools_node)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add edges
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )
        workflow.add_edge("tools", "agent")
        
        return workflow
    
    def _agent_node(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """
        Agent reasoning node - decides what to do next.
        
        Args:
            state: Current agent state
            config: Runnable configuration
        
        Returns:
            Updated state with agent's response
        """
        # Inject system prompt if this is the first message
        messages = state["messages"]
        if len(messages) == 1 or not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=ECOHOME_SYSTEM_PROMPT)] + messages
        
        # Get response from LLM
        response = self.llm_with_tools.invoke(messages, config)
        
        return {
            "messages": [response],
            "next_step": "continue" if response.tool_calls else "end"
        }
    
    def _tools_node(self, state: AgentState, config: RunnableConfig) -> AgentState:
        """
        Tools execution node - runs requested tools.
        
        Args:
            state: Current agent state
            config: Runnable configuration
        
        Returns:
            Updated state with tool results
        """
        # Get the last message (should have tool calls)
        last_message = state["messages"][-1]
        
        # Execute all tool calls
        tool_results = []
        for tool_call in last_message.tool_calls:
            # Find the tool
            tool = next(
                (t for t in ECOHOME_TOOLS if t.name == tool_call["name"]),
                None
            )
            
            if tool:
                try:
                    # Execute the tool
                    result = tool.invoke(tool_call["args"], config)
                    tool_results.append({
                        "tool_call_id": tool_call["id"],
                        "name": tool_call["name"],
                        "content": result
                    })
                except Exception as e:
                    tool_results.append({
                        "tool_call_id": tool_call["id"],
                        "name": tool_call["name"],
                        "content": f"Error executing tool: {str(e)}"
                    })
            else:
                tool_results.append({
                    "tool_call_id": tool_call["id"],
                    "name": tool_call["name"],
                    "content": f"Tool '{tool_call['name']}' not found"
                })
        
        # Create tool message
        from langchain_core.messages import ToolMessage
        tool_messages = [
            ToolMessage(
                content=result["content"],
                tool_call_id=result["tool_call_id"]
            )
            for result in tool_results
        ]
        
        return {"messages": tool_messages, "next_step": "continue"}
    
    def _should_continue(self, state: AgentState) -> Literal["continue", "end"]:
        """
        Determine if the agent should continue or end.
        
        Args:
            state: Current agent state
        
        Returns:
            "continue" to execute tools, "end" to finish
        """
        last_message = state["messages"][-1]
        
        # If there are tool calls, continue to tools node
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        
        return "end"
    
    def chat(self, message: str, thread_id: str = "default") -> str:
        """
        Send a message to the agent and get a response.
        
        Args:
            message: User message
            thread_id: Conversation thread ID for persistence
        
        Returns:
            Agent's response as string
        """
        # Create config with thread ID
        config = {"configurable": {"thread_id": thread_id}}
        
        # Create input state
        input_state = {
            "messages": [HumanMessage(content=message)],
            "next_step": "continue"
        }
        
        # Run the agent
        result = self.app.invoke(input_state, config)
        
        # Extract the final response
        final_message = result["messages"][-1]
        return final_message.content
    
    def stream_chat(self, message: str, thread_id: str = "default"):
        """
        Stream a conversation with the agent.
        
        Args:
            message: User message
            thread_id: Conversation thread ID for persistence
        
        Yields:
            Agent responses as they are generated
        """
        # Create config with thread ID
        config = {"configurable": {"thread_id": thread_id}}
        
        # Create input state
        input_state = {
            "messages": [HumanMessage(content=message)],
            "next_step": "continue"
        }
        
        # Stream the agent
        for event in self.app.stream(input_state, config):
            for node_name, node_output in event.items():
                if node_name == "agent" and node_output.get("messages"):
                    message = node_output["messages"][-1]
                    if hasattr(message, "content") and message.content:
                        yield message.content


def create_agent(model_name: str = "gpt-4o-mini", temperature: float = 0.7) -> EcoHomeAgent:
    """
    Factory function to create an EcoHome agent.
    
    Args:
        model_name: OpenAI model to use
        temperature: Temperature for LLM responses
    
    Returns:
        Configured EcoHomeAgent instance
    """
    return EcoHomeAgent(model_name=model_name, temperature=temperature)


if __name__ == "__main__":
    # Test the agent
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in .env file")
        exit(1)
    
    print("EcoHome Energy Advisor Agent")
    print("=" * 50)
    print("Type 'quit' to exit\n")
    
    # Create agent
    agent = create_agent()
    
    # Interactive loop
    thread_id = "test_session"
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        print("\nEcoHome: ", end="", flush=True)
        
        try:
            # Stream response
            for chunk in agent.stream_chat(user_input, thread_id):
                if chunk:
                    print(chunk, end="", flush=True)
            print("\n")
        except Exception as e:
            print(f"Error: {e}\n")
