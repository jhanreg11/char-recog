var clickX = new Array()
var clickY = new Array()
var clickDrag = new Array()
var paint
var context = $('#drawing-field')[0].getContext('2d')

function addClick(x, y, dragging) {
  clickX.push(x);
  clickY.push(y);
  clickDrag.push(dragging);
}

function redraw() {
  context.clearRect(0, 0, context.canvas.width, context.canvas.height); // Clears the canvas

  context.strokeStyle = "#df4b26";
  context.lineJoin = "round";
  context.lineWidth = 10;

  for(var i=0; i < clickX.length; i++) {
    context.beginPath();
    if(clickDrag[i] && i){
      context.moveTo(clickX[i-1], clickY[i-1]);
     }else{
       context.moveTo(clickX[i]-1, clickY[i]);
     }
     context.lineTo(clickX[i], clickY[i]);
     context.closePath();
     context.stroke();
  }
}

$(document).ready(function () {

  $('#drawing-field').mousedown(function(e){
    var mouseX = e.pageX - this.offsetLeft;
    var mouseY = e.pageY - this.offsetTop;

    paint = true;
    addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop);
    redraw();
  })

  $('#drawing-field').mousemove(function(e){
    if (paint) {
      addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop, true)
      redraw()
    }
  })

  $('#drawing-field').mouseup(function(e){
    paint = false

  })

  $('#drawing-field').mouseleave(function(e){
    paint = false;
  })

})
