// 이미지 입력 요소와 결과 출력 요소.
const imageInput = document.getElementById('imageInput');
const outputDiv = document.getElementById('output');

// 클래스 레이블을 정의.
const fruitLabels = {
    0: 'apple fruit',
    1: 'banana fruit',
    2: 'cherry fruit',
    3: 'chickoo fruit',
    4: 'grapes fruit',
    5: 'kiwi fruit',
    6: 'mango fruit',
    7: 'orange fruit',
    8: 'strawberry fruit'
};

// 모델 로드
let model;
(async function() {
    model = await tf.loadLayersModel('model.json');
    console.log('모델 로드 완료');
})();

// 이미지 분류 함수
async function classifyFruit() {
    const file = imageInput.files[0];
    
    if (!file) {
        alert('이미지 파일을 선택하세요.');
        return;
    }

    const img = await loadImage(file);
    const tensor = preprocessImage(img);

    // 모델을 사용하여 이미지 분류
    const predictions = await model.predict(tensor).data();

    // 예측된 클래스 인덱스
    const predictedClassIndex = predictions.indexOf(Math.max(...predictions));

    // 클래스 레이블
    const predictedLabel = fruitLabels[predictedClassIndex];

    // 결과 출력
    outputDiv.innerHTML = `<p>예측된 과일: ${predictedLabel}</p>`;
}

// 파일 선택이 변경될 때 이미지 분류 함수를 호출
imageInput.addEventListener('change', classifyFruit);

// 이미지 로드 함수
function loadImage(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = function() {
            const img = new Image();
            img.onload = function() {
                resolve(img);
            };
            img.src = reader.result;
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

// 이미지 전처리 함수 (크기 조정과 모델의 입력 형태에 맞게 조정)
function preprocessImage(img) {
    const tensor = tf.browser.fromPixels(img)
        .resizeNearestNeighbor([64, 64]) // 모델의 입력 크기에 맞게 조정 (64x64로 수정)
        .toFloat()
        .div(255.0) // 이미지 정규화 (0-1 범위로 변환)
        .expandDims();
    return tensor;
}
