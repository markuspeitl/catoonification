const express = require('express')
const exphbs  = require('express-handlebars');
//const formidableMiddleware = require('express-formidable')
const formidable = require('formidable');
const app = express();
var server = require('http').createServer(app);
const socketio = require('socket.io')(server);
const spawnChild = require("child_process").spawn;
const pathtools = require('path');

const imgsuploaddir = __dirname + '/uploadedimages'

app.engine('handlebars', exphbs({defaultLayout: 'main'}));
app.set('view engine', 'handlebars');

app.get('/', function (req, res) {
    res.render('home');
});

app.get('/home', function (req, res) {
    res.render('home');
});

app.use('/uploadedimages', express.static('uploadedimages'));
app.use(express.static(pathtools.join(__dirname, '/public')));
//app.use('/views',express.static('views'));

function stripExtension(path){
    var parsedPath = pathtools.parse(path)
    return parsedPath;
}

app.post('/', function (req, res) {
    //console.log(req.files)
    var form = new formidable.IncomingForm();
    form.uploadDir = imgsuploaddir;
    form
    .on('fileBegin',function(name,file){
        file.name = Date.now() + "_" + file.name;
        file.path = imgsuploaddir + "/" + file.name;
        console.log('fileBegin-' + name + ':' + JSON.stringify(file));
    })
    .on('progress',function(bytesReceived,bytesExpected){
        console.log('progress-' + bytesReceived +'/' + bytesExpected);
    })
    .on('aborted', function(){
        console.log('aborted');
    })
    .on('error', function(){
        console.log('error');
    })
    .on('end', function(){
        console.log('end');
    });

    form.parse(req,function(err,fields,files){
        //console.log('fields: ' + JSON.stringify(fields));
        //console.log('files: ' + JSON.stringify(files.filetoupload));
        var comment = fields.comment;
        var imageFile = files.filetoupload;
        if (imageFile){
            var name = imageFile.name;
            var path = imageFile.path;
            var type = imageFile.type;

            if(path){
                var pathmeta = stripExtension(name);
                var destname = pathmeta.name + "-dst" + pathmeta.ext;
                var destpath = imgsuploaddir + "/" + destname;
                console.log("spawning Python script")
                const pythonProcess = spawnChild('python3',["cartoonifyme.py", path, destpath ]);
                pythonProcess.stdout.on('data', (data) => {
                    console.log(data.toString('utf8'));
                });
                pythonProcess.stdout.on('close', (code) => {
                    //if(code === 1){
                    console.log("back in node, python exit detected")

                    let imagelink = "uploadedimages/" + destname;
                    res.redirect("/home?img=" + imagelink);
                    //res.send(JSON.stringify({"imagelink":imagelink}));
                    //res.send("localhost:3000/uploadedimages/" destname);
                    //}
                });
            }

            

            /*if (type.indexOf('image') != -1) {
                var outputPath = __dirname + '/multipart/' + Date.now() + '_' + name;

                fs.rename(path, outputPath, function (error) {
                    res.redirect('/');
                });
            } else {
                fs.unlink(path, function (error) {
                    res.send(400);
                });
            }*/
        } else {
            res.send(404);
        }
    });
});
 
server.listen(3000, function () {
    console.log('express-handlebars example server listening on: 3000');
});