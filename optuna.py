import optuna


from model import train_results  # Cambia esto por tu función real


def objective(trial):

    learning_rate = trial.suggest_loguniform('lr0', 1e-5, 1e-1)  # Rango logarítmico
    batch_size = trial.suggest_int('batch', 16, 64)                   # Rango entero
    epochs = trial.suggest_int('epochs', 10, 50)                           # Rango entero
    

    mAP = train_results(learning_rate=learning_rate, batch_size=batch_size, epochs=epochs)
    
    return mAP  


study = optuna.create_study(direction='maximize')  # Queremos maximizar la mAP
study.optimize(objective, n_trials=50)             # Ejecutar 50 pruebas


best_params = study.best_params
print("Mejores hiperparámetros:", best_params)