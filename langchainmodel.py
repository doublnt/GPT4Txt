from langchain.llms import LlamaCpp, GPT4All
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


def main():
    template = """Question: {question}

    Answer: Let's think step by step."""

    prompt = PromptTemplate(template=template, input_variables=["question"])

    local_path = "./models/ggml-gpt4all-j-v1.3-groovy.bin"

    # Make sure the model path is correct for your system!
    # Callbacks support token-wise streaming
    # Verbose is required to pass to the callback manager

    # callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    # llm = LlamaCpp(
    #     model_path=local_path, callback_manager=callback_manager, verbose=True
    # )

    # Verbose is required to pass to the callback manager
    # llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)
    # If you want to use GPT4ALL_J model add the backend parameter
    callbacks = [StreamingStdOutCallbackHandler()]
    llm = GPT4All(model=local_path, backend='gptj',
                  callbacks=callbacks, verbose=True)

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    question = "who is su shi first wife?"
    llm_chain.run(question)


if __name__ == "__main__":
    main()
