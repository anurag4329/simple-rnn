import h5py
import json
from tensorflow.keras.models import Sequential

# Step 1: Load model config from HDF5
with h5py.File('simple_rnn_imdb.h5', 'r') as f:
    model_config = json.loads(f.attrs['model_config'])

# Step 2: Recursively remove 'time_major' if present
def remove_time_major(config):
    if isinstance(config, dict):
        config.pop('time_major', None)
        for value in config.values():
            remove_time_major(value)
    elif isinstance(config, list):
        for item in config:
            remove_time_major(item)

remove_time_major(model_config)

# Step 3: Rebuild model from config
model = Sequential.from_config(model_config['config'])

# Step 4: Load weights from file
model.load_weights('simple_rnn_imdb.h5')

# Step 5: Save cleaned model
model.save('simple_rnn_imdb_cleaned.h5')
