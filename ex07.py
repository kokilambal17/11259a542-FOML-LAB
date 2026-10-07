# ============================================================
# SUPPRESS TENSORFLOW WARNINGS / LOG MESSAGES
# ============================================================

import os

# Disable TensorFlow oneDNN informational messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Reduce TensorFlow C++ log messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ============================================================
# Q-LEARNING
# ============================================================

import numpy as np

# Rewards matrix
R = np.array([
    [-1, -1, -1, -1,  0, -1],
    [-1, -1, -1,  0, -1, 100],
    [-1, -1, -1,  0, -1, -1],
    [-1,  0,  0, -1,  0, -1],
    [ 0, -1, -1,  0, -1, 100],
    [-1,  0, -1, -1,  0, 100]
])

# Initialize Q-table
Q = np.zeros_like(R, dtype=float)

gamma = 0.8
episodes = 1000

# Training
for _ in range(episodes):

    state = np.random.randint(0, 6)

    while state != 5:

        # Find valid actions
        actions = np.where(R[state] >= 0)[0]

        # Select random action
        action = np.random.choice(actions)

        # Next state
        next_state = action

        # Q-learning update
        Q[state, action] = (
            R[state, action]
            + gamma * np.max(Q[next_state])
        )

        state = next_state


# Display Q-table
print("Q-Table:\n")
print(Q)


# ============================================================
# NEURAL NETWORK
# ============================================================

import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# Reduce TensorFlow Python-level logging
tf.get_logger().setLevel("ERROR")


# ============================================================
# LOAD DATASET
# ============================================================

data = pd.read_csv("data.csv")


# Input features
X = data[['Hours_Studied', 'Attendance', 'Assignments']]

# Target variable
y = data['Result']


# Convert labels to numeric
y = y.map({
    'Fail': 0,
    'Pass': 1
})


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ============================================================
# BUILD NEURAL NETWORK
# ============================================================

model = tf.keras.Sequential([

    # Use Input() instead of input_shape
    # to remove the Keras warning
    tf.keras.Input(shape=(3,)),

    tf.keras.layers.Dense(
        8,
        activation='relu'
    ),

    tf.keras.layers.Dense(
        4,
        activation='relu'
    ),

    tf.keras.layers.Dense(
        1,
        activation='sigmoid'
    )
])


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# ============================================================
# TRAIN MODEL
# ============================================================

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    validation_split=0.2,
    verbose=1
)


# ============================================================
# PREDICT
# ============================================================

y_pred = model.predict(
    X_test,
    verbose=0
)


# Convert probabilities to class labels
y_pred = (y_pred > 0.5).astype(int)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:", accuracy)
