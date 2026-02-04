import torch
import torch.nn as nn

class WindForecaster(nn.Module):
    """
    LSTM Model to predict wind energy generation for the next 24 hours.
    Designed for GPU acceleration.
    """
    def __init__(self, input_size=3, hidden_size=128, num_layers=2, output_size=24):
        super(WindForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully Connected Layer to map to output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM forward
        out, _ = self.lstm(x)

        # Take the output from the last time step
        out = out[:, -1, :]

        # Pass through fully connected layer
        out = self.fc(out)
        return out
