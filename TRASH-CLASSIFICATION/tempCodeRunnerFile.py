batch_size = 38
epochs = 80
model.compile(optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# model.summary()

num_cardboard_train = len(os.listdir(imagepath_cardboard_dir))
num_glass_train = len(os.listdir(imagepath_glass_dir))
num_metal_train = len(os.listdir(imagepath_metal_dir))                  
num_paper_train = len(os.listdir(imagepath_cardboard_dir))
num_plastic_train = len(os.listdir(imagepath_glass_dir))
num_trash_train = len(os.listdir(imagepath_trash_dir))

num_cardboard_test = len(os.listdir(graypath_cardboard))
num_glass_test = len(os.listdir(graypath_glass))
num_metal_test = len(os.listdir(graypath_metal))
num_paper_test = len(os.listdir(graypath_paper))
num_plastic_test = len(os.listdir(graypath_plastic))
num_trash_test = len(os.listdir(graypath_trash))

total_train = num_cardboard_train + num_glass_train + num_metal_train + num_paper_train + num_plastic_train + num_trash_train
total_test = num_cardboard_test + num_glass_test + num_metal_test + num_paper_test + num_plastic_test + num_trash_test

print(total_train)
history = model.fit(
        train_data_gen,
        validation_data = train_data_gen,
        steps_per_epoch = total_train // batch_size,
        epochs = epochs,
        validation_steps= total_test // batch_size,
        callbacks = [tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    min_delta=0.01,
                    patience=7)]
    )                   