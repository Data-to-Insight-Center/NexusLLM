from patra_agent.graph import run_patra_graph

def main():
    # question = ("In the database, there are 3 models related to 3 deployments, Are there any missing models in the database? ")
    # question = ("In the database, Are all the models connected to a ModelCard? ")
    question = ("Create a connection from model with elementID = 4:b6ae30eb-5fdd-4c39-b281-fa2550f0ea84:3 to deployment with elementId = 4:b6ae30eb-5fdd-4c39-b281-fa2550f0ea84:11")
    # question = ("Is there a relationship between ModelCard and Deployment ? If not, Can you create a connection from this ModelCard to the mentioned Deployment?")
    
    result = run_patra_graph(question)
    print(result)

if __name__ == '__main__':
    main()