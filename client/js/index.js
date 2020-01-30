$(document).ready(function() {
  let canvas = document.getElementById('drawing-field')
  let ctx = canvas.getContext('2d')

  let painting = document.getElementById('paint');
  let paint_style = getComputedStyle(painting);
  canvas.width = parseInt(paint_style.getPropertyValue('width'));
  canvas.height = parseInt(paint_style.getPropertyValue('height'));

  let mouse = {x: 0, y: 0};

  canvas.addEventListener('mousemove', function(e) {
    mouse.x = e.pageX - this.offsetLeft;
    mouse.y = e.pageY - this.offsetTop;
  }, false);

  ctx.lineWidth = 3;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.strokeStyle = '#00CC99';

  canvas.addEventListener('mousedown', function(e) {
      ctx.beginPath();
      ctx.moveTo(mouse.x, mouse.y);

      canvas.addEventListener('mousemove', onPaint, false);
  }, false);

  canvas.addEventListener('mouseup', function() {
      canvas.removeEventListener('mousemove', onPaint, false);
  }, false);

  let onPaint = function() {
      ctx.lineTo(mouse.x, mouse.y);
      ctx.stroke();
  };

  function getChar() {

  }

  $('#predict').click(getChar)
})
