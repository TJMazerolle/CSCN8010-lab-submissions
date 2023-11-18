def neural_network_optimizer(x, y, x_val, y_val, batches, nodes, layers, activation_list = ["relu"], epochs = [10], output_activation_function = "softmax", 
                             optimizer = "adam", loss_function = "sparse_categorical_crossentropy", accuracy_metrics = ["accuracy"], cv = 1, verbose = True):

    """
    
    Limitations:
    * Only dense layers are implemented.
    * "adam" is the only optimizer used at this moment. However, it will not be difficult 
      to add more; it would just be an extra for loop going over different optimizers.
    * "softmax" is the only output activation function being tested since the function 
      was originally designed to perform multiclass classification.
    * "sparse_categorical_crossentropy" is the only loss function being tested since the 
      function was originally designed to perform multiclass classification.
    * This function does not implement proper cross-validation. It instead just reruns the
      prediction `cv` times and takes the average result without manually splitting the 
      training and validation sets.
    """

    categories = np.unique(y)

    batch_count = []
    node_count = []
    epoch_count = []
    activation_function = []
    layer_count = []
    accuracy = []

    for actfunc_index in range(len(activation_list)):
        for batch_index in range(len(batches)):
            for node_index in range(len(nodes)):
                for layer_index in range(len(layers)):
                    for epoch_index in range(len(epochs)):
                        for cv_index in range(cv):

                            accuracy_cv = []

                            # Initializing model
                            dense_nn = keras.Sequential()

                            # Adding first layer
                            dense_nn.add(Dense(nodes[node_index], activation = activation_list[actfunc_index], input_shape = (x.shape[1],)))

                            # Adding more layers
                            for num_layers in range(layers[layer_index] - 1):
                                dense_nn.add(Dense(nodes[node_index], activation = activation_list[actfunc_index]))
                            
                            # Adding output layer
                            dense_nn.add(Dense(len(categories), activation = output_activation_function))

                            # Compiling neural network
                            dense_nn.compile(optimizer = optimizer, loss = loss_function, metrics = accuracy_metrics)

                            # Training the model
                            fitted_nn = dense_nn.fit(x, y, batch_size = batches[batch_index], epochs = epochs[epoch_index], 
                                                     validation_data = (x_val, y_val), verbose = 0)

                            # Storing accuracy result
                            accuracy_cv.append(fitted_nn.history["val_accuracy"][len(fitted_nn.history["val_accuracy"]) - 1])

                        activation_function.append(activation_list[actfunc_index])
                        batch_count.append(batches[batch_index])
                        node_count.append(nodes[node_index])
                        layer_count.append(layers[layer_index])
                        epoch_count.append(epochs[epoch_index])
                        accuracy.append(np.mean(accuracy_cv))

                        if verbose:
                            print("Activation Function Used:", activation_list[actfunc_index], ",", actfunc_index + 1, "/", len(activation_list), "complete")
                            print("Batches Used:", batches[batch_index], ",", batch_index + 1, "/", len(batches), "complete")
                            print("Nodes Used:", nodes[node_index], ",", node_index + 1, "/", len(nodes), "complete")
                            print("Layers Used:", layers[layer_index], ",", layer_index + 1, "/", len(layers), "complete")
                            print("Epochs Used:", epochs[epoch_index], ",", epoch_index + 1, "/", len(epochs), "complete")
                            print("Accuracy:", fitted_nn.history["val_accuracy"][len(fitted_nn.history["val_accuracy"]) - 1])
                            print()

    results = pd.DataFrame()
    results["ActivationFunction"] = activation_function
    results["Layers"] = layer_count
    results["Batches"] = batch_count
    results["Nodes"] = node_count
    results["Epochs"] = epoch_count
    results["Accuracy"] = accuracy
    return results