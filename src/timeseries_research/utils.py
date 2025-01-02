import torch
import pandas as pd

def combine_backtest_split(backtest_split):
    """Combine the input, output and cutoff data into a single dataframe.

    Parameters
    ----------
    backtest_split : list
        A list of three dataframes:
        - First dataframe contains the cutoff dates
        - Second dataframe contains the input data
        - Third dataframe contains the output data

    Returns
    -------
    pandas.DataFrame
        A dataframe containing the combined input, output and cutoff data
    """
    cutoff = backtest_split[0].iloc[0,1]
    input = backtest_split[1].assign(cutoff=cutoff).assign(data_type='input')
    output = backtest_split[2].assign(cutoff=cutoff).assign(data_type='output')
    return pd.concat([input, output])


def create_seq2seq_dataset(df, input_size=180, output_size=14):
    """Convert dataframe into sequence-to-sequence PyTorch dataset

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns ['unique_id', 'ds', 'y', 'cutoff', 'data_type']
    input_size : int, default=180
        Length of input sequence
    output_size : int, default=14
        Length of target sequence

    Returns
    -------
    tuple
        inputs : torch.Tensor
            Tensor of shape (n_sequences, 1, input_size)
        targets : torch.Tensor 
            Tensor of shape (n_sequences, 1, output_size)
    """
    sequences = []
    targets = []
    
    for unique_id in df['unique_id'].unique():
        series_data = df[df['unique_id'] == unique_id]
        for cutoff in series_data['cutoff'].unique():
            window_data = df[df['cutoff'] == cutoff].sort_values('ds')
            
            # Split into input and target sequences
            input_data = window_data[window_data['data_type'] == 'input']['y'].values[-input_size:]
            target_data = window_data[window_data['data_type'] == 'output']['y'].values[:output_size]
            
            # Only keep sequences with full length
            if len(input_data) == input_size and len(target_data) == output_size:
                sequences.append(input_data)
                targets.append(target_data)
        
    # Convert to tensors
    inputs = torch.FloatTensor(sequences).unsqueeze(1)  # Add channel dimension
    targets = torch.FloatTensor(targets).unsqueeze(1)   # Add channel dimension
    
    return inputs, targets