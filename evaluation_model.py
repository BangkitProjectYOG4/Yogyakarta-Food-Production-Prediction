def evaluation(model, test_data, test_labels):
    __, mae, __ = model.evaluate(test_data, test_labels, verbose=0)
    print("Test Set Mean Abs Error : {:5.2f} Comodity in ton".format(mae))
