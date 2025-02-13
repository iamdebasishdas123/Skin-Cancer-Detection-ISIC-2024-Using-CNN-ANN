import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import EfficientNetB0

# Assuming X_train_tabular is already defined
# Input layers
image_input = Input(shape=(128, 128, 3), name='image_input')
tabular_input = Input(shape=(X_train_tabular.shape[1],), name='tabular_input')

# CNN for image data
effnet = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
effnet.trainable = False  # Freeze the pre-trained layers
x = effnet(image_input)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

# ANN for tabular data
y = Dense(128, activation='relu', kernel_initializer=GlorotNormal())(tabular_input)
y = Dropout(0.5)(y)
y = Dense(64, activation="relu", kernel_initializer=GlorotNormal())(y)
y = Dropout(0.3)(y)
y = Dense(32, activation="relu", kernel_initializer=GlorotNormal())(y)

# Concatenate both models
combined = Concatenate()([x, y])

# Final layers
z = Dense(64, activation='relu')(combined)
z = Dropout(0.5)(z)
output = Dense(1, activation='sigmoid')(z)

# Create the model
model = Model(inputs=[image_input, tabular_input], outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Build the model explicitly (VERY IMPORTANT)
model([tf.zeros((1, 128, 128, 3)), tf.zeros((1, X_train_tabular.shape[1]))])  # Pass dummy data

# Summary of the model
model.summary()

callback = EarlyStopping(monitor='loss', patience=5)

# Train the model
history = model.fit(
    [X_train_images, X_train_tabular], y_train,
    validation_data=([X_test_images, X_test_tabular], y_test),
    epochs=30,
    batch_size=16,
    callbacks=[callback]
)