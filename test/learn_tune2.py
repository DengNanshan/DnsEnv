def train_mnist(config, reporter):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (Dense, Dropout, Flatten, Conv2D,
                                         MaxPooling2D)
    batch_size = 128
    num_classes = 10
    epochs = 12
	#读取数据
    x_train, y_train, x_test, y_test, input_shape = get_mnist_data()
	#网络结构定义
    model = Sequential()
    model.add(
        Conv2D(
            32, kernel_size=(3, 3), activation="relu",
            input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(config["hidden"], activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
	#网络模型生成
    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.SGD(
            lr=config["lr"], momentum=config["momentum"]),
        metrics=["accuracy"])
	#网络模型训练
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        validation_data=(x_test, y_test),
        callbacks=[TuneReporterCallback(reporter)])

    if __name__ == "__main__":
        import ray
        from ray import tune
        from ray.tune.schedulers import AsyncHyperBandScheduler
        # 加载数据（上文def中没有具体加载数据）
        mnist.load_data()  # we do this on the driver because it's not threadsafe
        # 初始化ray
        ray.init()
        # 选择超参优化器及指定其参数
        sched = AsyncHyperBandScheduler(
            time_attr="training_iteration",
            metric="mean_accuracy",
            mode="max",
            max_t=400,
            grace_period=20)
        # 执行run程序，并且同时指定各种参数（可选）其中tune.run返回值为result，可用analysis接受以便后续分析
        analysis = tune.run(
            train_mnist,  # 先前定义的网络结构
            name="exp",  # experiment的名称，与输出结果后的路径有关
            scheduler=sched,  # 指定超参优化器
            stop={  # 设定提前终止trail的条件
                "mean_accuracy": 0.99,
                "training_iteration": 5 if args.smoke_test else 300
            },
            num_samples=10,  # 总共运行Trails的数目
            resources_per_trial={  # 每个Trail可支配的计算资源
                "cpu": 2,
                "gpu": 0
            },
            config={  # 设定超参空间
                "threads": 2,
                "lr": tune.sample_from(lambda spec: np.random.uniform(0.001, 0.1)),
                "momentum": tune.sample_from(
                    lambda spec: np.random.uniform(0.1, 0.9)),
                "hidden": tune.sample_from(
                    lambda spec: np.random.randint(32, 512)),
            })
        # （训练数据会自动保存至local_dir下，如："~/ray_results/exp"后面附录中有关于结果数据的详细讲解）
        # 输出最佳结果
        print("Best config is:", analysis.get_best_config(metric="mean_accuracy"))
        # ray.shutdown()——程序没有中断时，若需要再次ray.init()，则需要手动停止ray,否则报错