# VALID_LINK_CONSTRAINTS
# 
# This map defines all conceptually valid, directed relationships (A -> B)
# allowed in your graph based on the provided schema image.
# 
# Format: {SOURCE_NODE_LABEL: [ALLOWED_TARGET_NODE_LABEL, ...]}

VALID_LINK_CONSTRAINTS = {
    'Model Card': ['Data Sheet', 'Model Requirements', 'Bias Analysis', 'Explainability Analysis', 'Model'],

    'Model': ['Deployment', 'Experiment'],

    'Server' : ['Deployment'],

    'Deployment': ['Experiment', 'Device'],

    'Experiment': ['Raw Image', 'User', 'Device', 'Model'],
    
    'Data Sheet': ['Model Card'],
    
    'Model Requirements': ['Model Card'],
    
    'Bias Analysis': ['Model Card'],
    
    'Explainability Analysis': ['Model Card'],

    'User': ['Experiment'],

    'Raw Image': ['Experiment'],

    'Device': ['Deployment', 'Experiment']
}