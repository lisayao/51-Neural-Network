
window.addEventListener('load', eventWindowLoaded, false);  
function eventWindowLoaded() {
    canvasApp();
}



function canvasApp(){  
var theCanvas = document.getElementById('canvas');
var context = document.getElementById('canvas').getContext("2d");
var resetButton = document.getElementById("reset_image");
resetButton.addEventListener('click', resetPressed, false);
drawScreen();
}

    function drawScreen() {
        document.getElementById('canvas').addEventListener('mousedown', mouse_pressed_down, false);
        document.getElementById('canvas').addEventListener('mousemove', mouse_moved, false);
        document.getElementById('canvas').addEventListener('mouseup', mouse_released, false);
        document.getElementById('canvas').addEventListener('touchmove', touch_move_gesture, false);
        document.getElementById('canvas').getContext("2d").fillStyle = 'white';
        document.getElementById('canvas').getContext("2d").fillRect(0, 0, document.getElementById('canvas').width, document.getElementById('canvas').height);
        document.getElementById('canvas').getContext("2d").strokeStyle = '#000000';
        document.getElementById('canvas').getContext("2d").strokeRect(1,  1, document.getElementById('canvas').width-2, document.getElementById('canvas').height-2);
    }

// For the mouse_moved event handler.
var begin_drawing = false;

function mouse_pressed_down (ev) {
    begin_drawing = true;
    document.getElementById('canvas').getContext("2d").fillStyle = "#000000";
}

function mouse_moved (ev) {
    var x, y;
    // Get the mouse position in the canvas
    x = ev.pageX;
    y = ev.pageY;

    if (begin_drawing) {
        document.getElementById('canvas').getContext("2d").beginPath();
        document.getElementById('canvas').getContext("2d").arc(x, y, 7, (Math.PI/180)*0, (Math.PI/180)*360, false);
        document.getElementById('canvas').getContext("2d").fill();
        document.getElementById('canvas').getContext("2d").closePath();
    }
}

function mouse_released (ev) {
    begin_drawing = false;
}

function touch_move_gesture (ev) {
    // For touchscreen browsers/readers that support touchmove
    var x, y;
    context.beginPath();
    context.fillStyle = colorChosen.innerHTML;
    if(ev.touches.length == 1){
        var touch = ev.touches[0];
        x = touch.pageX;
        y = touch.pageY;
        context.arc(x, y, 7, (Math.PI/180)*0, (Math.PI/180)*360, false);
        context.fill();
    }
}


function resetPressed(e) {
    document.getElementById('canvas').width = document.getElementById('canvas').width; // Reset grid
    drawScreen();
}
