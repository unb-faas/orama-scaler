from sklearn.model_selection import train_test_split

def divide(data):
    X = data.drop(columns=['Latency'], axis=1)
    y = data['Latency']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("........... X Train ............")
    print(X_train)
    print("........... Y Train ............")
    print(y_train)
    print("........... X Test ............")
    print(X_test) 
    print("........... Y Test ............")
    print(y_test) 
    return X_train, X_test, y_train, y_test

