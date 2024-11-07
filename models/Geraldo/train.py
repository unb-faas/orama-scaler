def fit(X_train, y_train, model, dir):
    train_results = model.fit(X_train, y_train, epochs=10, batch_size=500, validation_split=0.3, verbose=1)
    model.save(f'{dir}/model.keras')
    return train_results, model