import torch
import pandas as pd
from utilsforecast.processing import backtest_splits

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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def auto_backtest_splits(y_df, h=52, freq='D', id_col='unique_id', time_col='ds', step_size=1, input_size=104):
    """Automatically determine number of windows for backtesting splits.
    Calculates the maximum number of windows possible given the data length,
    horizon, input size and step size, then performs backtesting splits.

    Parameters
    ----------
    y_df : pandas.DataFrame
        DataFrame containing the time series data
    h : int, optional
        Forecast horizon. Defaults to 52.
    freq : str, optional
        Frequency of the time series. Defaults to 'D'.
    id_col : str, optional
        Name of ID column. Defaults to 'unique_id'.
    time_col : str, optional
        Name of time column. Defaults to 'ds'.
    step_size : int, optional
        Number of steps between windows. Defaults to 1.
    input_size : int, optional
        Size of input sequence. Defaults to 104.

    Returns
    -------
    Generator[Tuple[DataFrame, DataFrame, DataFrame]]
        Generator yielding tuples of (cutoffs, train, valid) for each window.
    """
    combined = []
    for id in y_df.unique_id.unique():
        sub_y_df = y_df.query('unique_id == @id')
        n_windows = int(sub_y_df.shape[0] - (h + input_size) / step_size)
        
        out = backtest_splits(
            sub_y_df, 
            n_windows=n_windows, 
            h=h, 
            freq=freq, 
            id_col=id_col, 
            time_col=time_col, 
            step_size=step_size, 
            input_size=input_size
        )
        
        # make a delta friendly format with the maximum number of windows
        combined = combined + [combine_backtest_split(x) for x in out]
        
    combined_df = pd.concat(combined)
    return combined_df
    