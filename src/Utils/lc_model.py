#region import
from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.prebuilt import tools_condition
from langchain_core.tools import StructuredTool

from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict
#endregion

#region static input vectorstore
## Retriever from input json
from langchain.tools.retriever import create_retriever_tool
def get_static_retriever():
      all_data_json_file = "input_data/all_data.json"
      loader = JSONLoader(file_path=all_data_json_file, jq_schema=".", text_content=False)
      collection = loader.load()
      text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
      chunk_size=100, chunk_overlap=50
      )
      doc_splits = text_splitter.split_documents(collection)
      # Add to vectorDB
      vectorstore = Chroma.from_documents(
      documents=doc_splits,
      collection_name="rag-chroma",
      embedding=OpenAIEmbeddings(),
      )
      retriever = vectorstore.as_retriever()
      retriever_tool_rag = create_retriever_tool(
            retriever,
            "retrieve_rag_input_information",
            "Search for information about buffets, coffe breaks, wines tasting, and hotel pricing policy. Note that the stored currency is €",
      ) 
      return retriever_tool_rag


#endregion

#region boiler plate

from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]

#endregion

#region node and edges

#region Edges
def question_router(state) -> Literal["agent", "normal_agent"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    # model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", streaming=True)
    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing whether the user question is related to hotel booking, buffet, coffe, wine or similar aspect or not. \n 
        Here is the user question: {question} \n
        If the user question contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    # docs = last_message.content

    scored_result = chain.invoke({"question": question})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: QUESTION RELETED TO BOOKING AND FOOD---")
        return "agent"

    else:
        print("---DECISION: QUESTION NOT RELETED TO BOOKING AND FOOD---")
        print(score)
        return "normal_agent"

#endregion

#region Nodes
from functools import partial
# def get_agent_with_tools(tools):
#      return partial(agent, tools = tools)

# def agent(state):
#     """
#     Invokes the agent model to generate a response based on the current state. Given
#     the question, it will decide to retrieve using the retriever tool, or simply end.

#     Args:
#         state (messages): The current state

#     Returns:
#         dict: The updated state with the agent response appended to messages
#     """
#     print("---CALL AGENT---")
#     messages = state["messages"]
#     # model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4-turbo")
#     model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4o-mini")
#     model = model.bind_tools(tools)
#     msg = [
#         HumanMessage(
#             content=f""" \n 
#     Look at the initial question and the currently retrieved information in the conversation, make a summary of all the retrieved information or call the tool to retrieved more information if there is something needed to answer the question but have not been called.
#     \n
#     Here is the conversation:
#     \n ------- \n
#     {messages} 
#     \n ------- \n
#     """
#     )]

#     response = model.invoke(msg)
#     # We return a list, because this will get added to the existing list
#     return {"messages": [response]}


def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[-2].content

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning.
    Split the question into small question easier to answear, e.g question about price of booking a hotel in a period with breakfast and event is splited into 3 question about hotel price in a period, breakfast price, and event price.
    \n
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    # Grader
    # model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)
    model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4o-mini")
    
    response = model.invoke(msg)
    return {"messages": [response]}

def normal_agent(state):
    """
    Normal chat for topic unrelated to booking and food
    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL NORMAL AGENT---")
    messages = [state["messages"][0]]
    model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4o-mini")
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    # Prompt
    # prompt = hub.pull("rlm/rag-prompt")
    prompt = ChatPromptTemplate([
        # ("system", ""),
        ("human", 
"""You are an assistant for helping retrieve information and calculate the total price for booking. Use the following pieces of retrieved context and price to calculate the total price the booking in question.
If something is unavailable, say that it is unavailable or you can't find information about it and omit it from the total price. 
Keep the answer concise and summary the information instead of write it all out unless asked to.
If reasonable, provide your answer which include a sentence stating the total cost of this format: 'The cost for your stay is $500, including breakfast and a wellness package.', also provide a short summary of the cost of each component unless the user asked not to provide it. 
Remember to convert different currency type into 1 currency type before doing the calculation.
Question: {question} 
Context: {context} 
Answer:""")
    ])
    # You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    # Question: {question} 
    # Context: {context} 
    # Answer:

    # LLM
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)
    llm = ChatOpenAI(temperature=0, streaming=True, model="gpt-4o-mini")

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}


#endregion

#endregion

#region display
# from IPython.display import Image, display

# try:
#     display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
# except Exception:
#     # This requires some extra dependencies and is optional
#     pass
#endregion

#endregion

#region Usage

#region graph
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

from . import hotel_web_crawl
def get_lc_graph():
      price_browser_tool = StructuredTool.from_function(
           func=hotel_web_crawl.get_price_by_dates,
           name = "hotel_price_browser_tool",
           description=hotel_web_crawl.get_price_by_dates.__doc__,
        )
      tools = [get_static_retriever(), price_browser_tool]
      retrieve = ToolNode(tools)
      #region test
      
      ### NOTE: put this function into here as a lazy solution to avoid having to pre-load it with tools for binding
      def agent(state):
            """
            Invokes the agent model to generate a response based on the current state. Given
            the question, it will decide to retrieve using the retriever tool, or simply end.

            Args:
                  state (messages): The current state

            Returns:
                  dict: The updated state with the agent response appended to messages
            """
            print("---CALL AGENT---")
            messages = state["messages"]
            # model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4-turbo")
            model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4o-mini")
            ## cap recursion at 12, not binding tools when exceeded
            if len(messages) < 12:
                model = model.bind_tools(tools)
                msg = [
                    HumanMessage(
                            content=f""" \n 
                Look at the initial question and the currently retrieved information in the conversation, 
                make a summary of all the retrieved information or call the tool to retrieved more information if there is something needed to answer the question but have not been called.
                \n
                Here is the conversation:
                \n ------- \n
                {messages} 
                \n ------- \n
                """
                )]

            
                response = model.invoke(msg)
            else:
                capped_message = [
                    HumanMessage(
                            content=f""" \n 
                Look at the initial question and the currently retrieved information in the conversation, 
                make a summary of all the retrieved information, or call the tool to retrieved more information if there is something needed to answer the question but have not been called.
                \n
                Here is the conversation:
                \n ------- \n
                {messages} 
                \n ------- \n
                """
                )]
                response = model.invoke(capped_message)
            print("respond of agent: ", response)
            # We return a list, because this will get added to the existing list
            return {"messages": [response]}



      #endregion

      # Define a new graph
      workflow = StateGraph(AgentState)
      #region node
      # Define the nodes we will cycle between
      workflow.add_node("agent", agent)  # agent
      workflow.add_node("normal_agent", normal_agent)
      workflow.add_node("retrieve", retrieve)  # retrieval
    #   workflow.add_node("rewrite", rewrite)  # Re-writing the question
      workflow.add_node("generate", generate)  # Generating a response after we know the documents are relevant
      #endregion
      #region edges
      # Call agent node to decide to retrieve or not
      workflow.add_conditional_edges(
            START, 
            question_router
      ) ## question about booking or not

    #   workflow.add_edge(START, "agent")
      # Decide whether to retrieve
      workflow.add_conditional_edges(
      "agent",
      # Assess agent decision
      tools_condition,
      {
            # Translate the condition outputs to nodes in our graph
            "tools": "retrieve",
            END: "generate",
      },
      )

      # Edges taken after the `action` node is called.
    #   workflow.add_conditional_edges(
    #   "retrieve",
    #   # Assess agent decision
    #   grade_documents,
    #   )
      workflow.add_edge("retrieve", "agent")
      # workflow.add_edge("generate", END)
      workflow.add_edge("generate", END)
      workflow.add_edge("normal_agent", END)
    #   workflow.add_edge("rewrite", "agent")
      
      #endregion
      # Compile
      graph = workflow.compile()

      return graph

def get_respond_from_graph(message, graph):
      """
            Args:
                  string message, langgraph graph
            Return:
                  string respond
      """
      inputs = {"messages": [
            ("user", message),
      ]}
      message = graph.invoke(inputs)
      return message['messages'][-1].content

#endregion

#endregion