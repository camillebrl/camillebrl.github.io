document.addEventListener('mousemove', function(e) {
    var follower = document.getElementById('follower');
    if (follower) {
        follower.style.left = (e.pageX - 10) + 'px';
        follower.style.top = (e.pageY - 10) + 'px';
    }
});

let isDrawing = false;
let x = 0;
let y = 0;

const canvas = document.createElement('canvas');
canvas.style.position = 'absolute';
canvas.width = document.documentElement.scrollWidth;
canvas.height = document.documentElement.scrollHeight;
canvas.style.top = '0';
canvas.style.left = '0';
canvas.style.zIndex = '-2';
canvas.style.pointerEvents = 'none';
canvas.style.backgroundColor = 'transparent';
canvas.style.filter = 'blur(2px)';
document.body.appendChild(canvas);

const context = canvas.getContext('2d');
context.strokeStyle = 'red';
context.lineWidth = 5;
context.lineJoin = 'round';
context.lineCap = 'round';

document.addEventListener('mousedown', function(e) {
    isDrawing = true;
    x = e.pageX;
    y = e.pageY;
});

document.addEventListener('mousemove', function(e) {
    if (isDrawing) {
        drawLine(x, y, e.pageX, e.pageY);
        x = e.pageX;
        y = e.pageY;
    }
});

document.addEventListener('mouseup', function() {
    isDrawing = false;
});

function drawLine(x1, y1, x2, y2) {
    context.beginPath();
    context.moveTo(x1, y1);
    context.lineTo(x2, y2);
    context.stroke();
}

function fadeOutCanvas() {
    context.fillStyle = 'rgba(255, 255, 255, 0.1)';
    context.fillRect(0, 0, canvas.width, canvas.height);
    requestAnimationFrame(fadeOutCanvas);
}

fadeOutCanvas();
