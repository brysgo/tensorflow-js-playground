const tf = require("@tensorflow/tfjs"); // If not loading the script as a global

test("follow first simple example", async () => {
  const a = tf.tensor1d([1, 2, 3]);
  const b = tf.scalar(2);

  const result = a.add(b); // a is not modified, result is a new tensor
  expect(await result.data()).toEqual(new Float32Array([3, 4, 5]));

  // Alternatively you can use a blocking call to get the data.
  // However this might slow your program down if called repeatedly.
  expect(result.dataSync()).toEqual(new Float32Array([3, 4, 5]));
});

test("building a model for linear regression", async () => {
  // A sequential model is a container which you can add layers to.
  const model = tf.sequential();

  // Add a dense layer with 1 output unit.
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  // Specify the loss type and optimizer for training.
  model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

  // Generate some synthetic data for training.
  const xs = tf.tensor2d([[1], [2], [3], [4]], [4, 1]);
  const ys = tf.tensor2d([[1], [3], [5], [7]], [4, 1]);

  // Train the model.
  await model.fit(xs, ys, { epochs: 500 });

  // After the training, perform inference.
  const output = model.predict(tf.tensor2d([[5]], [1, 1]));
  expect(Math.round((await output.data())[0])).toEqual(9);
});
