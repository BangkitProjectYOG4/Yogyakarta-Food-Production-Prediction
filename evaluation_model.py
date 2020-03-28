import pandas as pd

def evaluation(model, test_data, test_labels):
    loss, mse = model.evaluate(test_data, test_labels, verbose=0)
    print("Mean Absolute Error : {:5.2f} Comodity in ton".format(loss))
    print("Mean Squared Error : {:5.2f} Comodity in ton".format(mse))
    predict = model.predict(test_data).flatten()

    predict = pd.DataFrame(list(zip(predict, test_labels.tolist())), 
                columns =['Prediction','Actual'])

    predict['error'] = (abs(predict['Actual'] - predict['Prediction'])/
                                predict['Prediction'])
    print("Mean Error Rate: {}\n".format(predict.error.mean()))
