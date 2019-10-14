var canvas, ctx,
  brush = {
    x: 0,
    y: 0,
    color: '#000000',
    size: 8,
    down: false
  },
  strokes = [],
  currentStroke = null

function redraw() {
  ctx.clearRect(0, 0, canvas.width(), canvas.height())
  ctx.lineCap = 'round'
  for (var i = 0; i < strokes.length; i++) {
    var s = strokes[i]
    ctx.strokeStyle = s.color
    ctx.lineWidth = s.size
    ctx.beginPath()
    ctx.moveTo(s.points[0].x, s.points[0].y)
    for (var j = 0; j < s.points.length; j++) {
      var p = s.points[j]
      ctx.lineTo(p.x, p.y)
    }
    ctx.stroke()
  }
}

function init() {
  canvas = $('#drawing-field')
  ctx = canvas[0].getContext('2d')
  function mouseEvent (e) {
    offset = $('#drawing-field').offset()
    brush.x = e.pageX - offset.left
    brush.y = e.pageY - offset.top
    currentStroke.points.push({
      x: brush.x,
      y: brush.y
    })

    redraw()
  }

  canvas.mousedown(function (e) {
    brush.down = true

    currentStroke = {
      color: brush.color,
      size: brush.size,
      points: []
    }

    strokes.push(currentStroke)

    mouseEvent(e)
  }).mouseup(function (e) {
    brush.down = false

    mouseEvent(e)

    currentStroke = null
  }).mousemove(function (e) {
    if (brush.down)
      mouseEvent(e)
  }).mouseleave(function (e) {
    brush.down = false
  })
}

$(init)
