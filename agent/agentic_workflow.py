from utils.model_loader import ModelLoader
from prompt_library.prompt import SYSTEM_PROMPT
from prompt_library.prompt_library import SYSTEM_PROMPT

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import tool_node, tools_condition

from tools.weather_tool import WeatherTool
from tools.place_search_tool import PlaceSearchTool
from tools.currency_conversion_tool import CurrencyConversionTool
from tools.calculator_tool import CalculatorTool
from tools.arithmatic_tool import ArithmeticTool


class GraphBuilder():
    
    def __init__(self,model_provider:str = "groq"):
        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()
        
        self.tools=[]
        self.weather_tool = WeatherTool()
        self.place_search_tool = PlaceSearchTool()
        self.currency_conversion_tool = CurrencyConversionTool()
        self.calculator_tool = CalculatorTool()
        
        self.llm_with_tools = self.llm.bind_tools(
            self.weather_tool,
            self.place_search_tool,
            self.currency_conversion_tool,
            self.calculator_tool       
            )
        
        self.system_prompt = SYSTEM_PROMPT
        self.graph = StateGraph()
    
    def agent_function(self,state:MessagesState):
        """Main agent function"""
        user_question=state["messages"]
        input_question=self.system_prompt + user_question
        response= self.llm_with_tools.invoke(input_question)
        return {"messages":[response]}
        
    def build_graph(self):
        graph_builder=StateGraph(MessagesState)
        graph_builder.add_node("agent",self.agent_function)
        graph_builder.add_node("tools", tool_node(self.llm_with_tools))
        graph_builder.add_edge(START, "agent")
        graph_builder.add_conditional_edges("agent", tools_condition)
        graph_builder.add_edge("tools", "agent")
        graph_builder.add_edge("agent", END)
        
        self.graph = graph_builder.compile()
        return self.graph
    
    def __call__(self):
        self.build_graph()