import sys

import tensorflow as tf
from absl import app

train_filename = "train.csv" #update model parameter
val_filename = "valid.csv" #estimate & monitor model quality
test_filename = "test.csv" # monitor model quality after model is trained 

learning_rate = 0.0002282433105027466
hidden_layer_dims = [768, 256, 128]
BATCH_SIZE = 512

num_train_steps = 150000
num_eval_steps = 8634

LABEL_FEATURE = "label"
LABEL_FEATURE_TYPE = tf.float32

NUMERIC_FEATURES = ["I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10", "I11", "I12", "I13"]
NUMERIC_FEATURE_TYPES = [tf.float32] * len(NUMERIC_FEATURES)

CATEGORICAL_FEATURES = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", "C15",
                        "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26"]
CATEGORICAL_FEATURE_TYPES = [tf.int32] * len(CATEGORICAL_FEATURES)
CATEGORICAL_FEATURE_EMBEDDING_DIMENSION = 20
NUM_BINS = 10000

#read csv file by batch and parallel reading in each batch 
# change epoch to 1 if is test set 
def get_dataset(file_pattern, is_test):
    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=file_pattern,
        batch_size=BATCH_SIZE,
        column_names=[LABEL_FEATURE] + NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        column_defaults=[LABEL_FEATURE_TYPE] + NUMERIC_FEATURE_TYPES + CATEGORICAL_FEATURE_TYPES,
        label_name=LABEL_FEATURE,
        header=True,
        num_epochs=1 if is_test else None, # number of path to test an example 
        shuffle=True,
        num_parallel_reads=16) # number of working thread on the same file 
    return dataset


class M(tf.keras.Model):

    def __init__(self, layer_dims, layer_activations):
        super(M, self).__init__()

        self.Es = {}
        self.Hs = {}

        # each Categorical feature -> HASH_layer -> EMBED_layer 
		# each int -> bucket between 0 - -1 (parameter # of bins) 
        for name in CATEGORICAL_FEATURES:
            self.Hs[name] = tf.keras.layers.Hashing(NUM_BINS)
            self.Es[name] = tf.keras.layers.Embedding(
                NUM_BINS, CATEGORICAL_FEATURE_EMBEDDING_DIMENSION)
        #DENSE Layer 
        self.Ls = []
        # non linear activation function to model complex relationship between input and output
        for dim, activation in zip(layer_dims, layer_activations):
            self.Ls += [tf.keras.layers.Dense(dim, activation=activation)]

    @tf.function
    def call(self, inputs):
        outputs = []
        for name in NUMERIC_FEATURES:
            output = tf.reshape(inputs[name], [-1, 1])
            outputs.append(output)
        for name in CATEGORICAL_FEATURES:
            output = self.Hs[name](inputs[name])
            output = self.Es[name](output)
            outputs.append(output)
        outputs = tf.keras.layers.concatenate(outputs)
        for L in self.Ls:
            outputs = L(outputs)
        #apply sigmoid (activation function) to generate overall output for binary classification, output binary probability 
        return tf.math.sigmoid(outputs)

def main(argv):
    del argv

    train_dataset = get_dataset(train_filename, False)
    # print("DATASET", next(train_dataset.take(1).as_numpy_iterator()))
    # return

    val_dataset = get_dataset(val_filename, False)
    test_dataset = get_dataset(test_filename, True)

    model = M(
        layer_dims=hidden_layer_dims + [1],
        #no activation function applied for the last layer 
        layer_activations=["relu"] * len(hidden_layer_dims) + [None],
    )
    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=learning_rate, clipnorm=100
    )
    model.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )

    history = model.fit(
        x=train_dataset,
        epochs=1500,
        steps_per_epoch=100,
        verbose=2,
        # validation_data=eval_dataset,
        # validation_steps=100,
    )

    results = model.evaluate(test_dataset)
    print('results', results)


if __name__ == "__main__":
    app.run(main)