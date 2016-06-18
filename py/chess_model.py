import model

def Model():
  return model.Model(input_dim=64, embedding_rows=13, embedding_cols=16, 
    output_dim=4096, hidden_dims=[2048, 2048, 2048], lr=0.01, 
    reg_factor=0.0001)
