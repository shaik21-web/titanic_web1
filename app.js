
let session;

async function loadModel() {
    session = await ort.InferenceSession.create("titanic_model.onnx");
    console.log("ONNX model loaded!");
}

function getInputVector() {
    return Float32Array.from([
        parseFloat(document.getElementById("pclass").value),
        parseFloat(document.getElementById("sex").value),
        parseFloat(document.getElementById("age").value),
        parseFloat(document.getElementById("sibsp").value),
        parseFloat(document.getElementById("parch").value),
        parseFloat(document.getElementById("fare").value),
        parseFloat(document.getElementById("embarked_c").value),
        parseFloat(document.getElementById("embarked_q").value)
    ]);
}

async function predict() {
    const inputVector = getInputVector();
    const inputTensor = new ort.Tensor("float32", inputVector, [1, inputVector.length]);

    const feeds = { input: inputTensor };
    const results = await session.run(feeds);
    const output = results.output.data[0];

    const prob = 1 / (1 + Math.exp(-output)); // sigmoid
    const survived = prob >= 0.5 ? "Yes" : "No";

    document.getElementById("result").innerHTML = 
        `Survival Probability: ${(prob*100).toFixed(2)}% <br>Survived? ${survived}`;
}

document.getElementById("predictBtn").addEventListener("click", predict);

loadModel();
