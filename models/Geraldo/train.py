def fit(X_train, y_train, X_test, y_test, model, dir):
    train_results = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.3, verbose=1, validation_data=(X_test, y_test))
    model.save(f'{dir}/model.keras')
    return train_results, model