request = new Request()

function clearDrawing() {
    var canvas = document.querySelector('#paint');
    var ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function submitDrawing() {
    var canvas = document.querySelector('#paint');
    let tmpCanvas = $('<canvas/>', {
        id: 'resizeCanvas'
    }).prop({
        width: 28,
        height: 28,
    })

    let newCanvas = $('<canvas>')
      .attr("width", 28)
      .attr("height", 28)

    let newContext = newCanvas[0].getContext('2d')
    newContext.scale(28 / canvas.width, 28 / canvas.height)
    newContext.drawImage(canvas, 0, 0)

    let imageData =newContext.getImageData(0, 0, 28, 28).data
	  let bluePixels = Array.prototype.slice.call(imageData.slice(784 * 2, 784 * 3))

    reqData = {pixels: bluePixels}
    request.POST(reqData, '/classify', function(result) {
        $('#result').html('Result: ' + result.result)
    })
    $('#result').html('Working...')
}

//Here is the main code for the paint window
$(document).ready(function() {
    
    var canvas = document.querySelector('#paint');
    var ctx = canvas.getContext('2d');
    
    var sketch = document.querySelector('#sketch');
    var sketch_style = getComputedStyle(sketch);
    canvas.width = parseInt(sketch_style.getPropertyValue('width'));
    canvas.height = parseInt(sketch_style.getPropertyValue('height'));
    
    
    // Creating a tmp canvas
    var tmp_canvas = document.createElement('canvas');
    var tmp_ctx = tmp_canvas.getContext('2d');
    tmp_canvas.id = 'tmp_canvas';
    tmp_canvas.width = canvas.width;
    tmp_canvas.height = canvas.height;
    
    sketch.appendChild(tmp_canvas);

    var mouse = {x: 0, y: 0};
    var last_mouse = {x: 0, y: 0};
    
    // Pencil Points
    var ppts = [];
    
    /* Mouse Capturing Work */
    tmp_canvas.addEventListener('mousemove', function(e) {
        mouse.x = typeof e.offsetX !== 'undefined' ? e.offsetX : e.layerX;
        mouse.y = typeof e.offsetY !== 'undefined' ? e.offsetY : e.layerY;
    }, false);
    
    
    /* Drawing on Paint App */
    tmp_ctx.lineWidth = 15;
    tmp_ctx.lineJoin = 'round';
    tmp_ctx.lineCap = 'round';
    tmp_ctx.strokeStyle = 'blue';
    tmp_ctx.fillStyle = 'blue';
    
    tmp_canvas.addEventListener('mousedown', function(e) {
        tmp_canvas.addEventListener('mousemove', onPaint, false);
        
        mouse.x = typeof e.offsetX !== 'undefined' ? e.offsetX : e.layerX;
        mouse.y = typeof e.offsetY !== 'undefined' ? e.offsetY : e.layerY;
        
        ppts.push({x: mouse.x, y: mouse.y});
        
        onPaint();
    }, false);
    
    tmp_canvas.addEventListener('mouseup', function() {
        tmp_canvas.removeEventListener('mousemove', onPaint, false);
        
        // Writing down to real canvas now
        ctx.drawImage(tmp_canvas, 0, 0);
        // Clearing tmp canvas
        tmp_ctx.clearRect(0, 0, tmp_canvas.width, tmp_canvas.height);
        
        // Emptying up Pencil Points
        ppts = [];
    }, false);
    
    var onPaint = function() {
        
        // Saving all the points in an array
        ppts.push({x: mouse.x, y: mouse.y});
        
        if (ppts.length < 3) {
            var b = ppts[0];
            tmp_ctx.beginPath();
            //ctx.moveTo(b.x, b.y);
            //ctx.lineTo(b.x+50, b.y+50);
            tmp_ctx.arc(b.x, b.y, tmp_ctx.lineWidth / 2, 0, Math.PI * 2, !0);
            tmp_ctx.fill();
            tmp_ctx.closePath();
            
            return;
        }
        
        // Tmp canvas is always cleared up before drawing.
        tmp_ctx.clearRect(0, 0, tmp_canvas.width, tmp_canvas.height);
        
        tmp_ctx.beginPath();
        tmp_ctx.moveTo(ppts[0].x, ppts[0].y);
        
        for (var i = 1; i < ppts.length - 2; i++) {
            var c = (ppts[i].x + ppts[i + 1].x) / 2;
            var d = (ppts[i].y + ppts[i + 1].y) / 2;
            
            tmp_ctx.quadraticCurveTo(ppts[i].x, ppts[i].y, c, d);
        }
        
        // For the last 2 points
        tmp_ctx.quadraticCurveTo(
            ppts[i].x,
            ppts[i].y,
            ppts[i + 1].x,
            ppts[i + 1].y
        );
        tmp_ctx.stroke();
        
    };

    $('#predict').click(submitDrawing)
    $('#clear').click(clearDrawing)
})