const express = require('express')
const exphbs  = require('express-handlebars');
//const formidableMiddleware = require('express-formidable')
const formidable = require('formidable');
const app = express();
var server = require('http').createServer(app);
const socketio = require('socket.io')(server);
var spawnChild = require("child_process").spawn;
const pathtools = require('path');
//const bincaller = require('./bincaller.js')
const filtermanager = require('./filtermanager.js')
const utilfns = require('./utilfns')

const imgsuploaddir = __dirname + '/uploadedimages'

const registeredFilters = ["trippy1","trippy2","directionglow","mosaic","net",
                            "noiser","colorblob","formsurfaceblobs","smallformsurfaceblobs",
                            "smallsurfaceblobs","pencil","colorpencil","pixelpencil","pixelart"]

const filterMappingDict = {
    "trippy1":"trippy1",
    "trippy2":"trippy2",
    "directionglow":"directionglow",
    "mosaic":"mosaic",
    "net":"net",
    "noiser":"noiser",
    "colorblob":"colorblob",
    "formsurfaceblobs":"formsurfaceblobs",
    "smallformsurfaceblobs":"smallformsurfaceblobs",
    "smallsurfaceblobs":"smallsurfaceblobs",
    "pencil":"pencil",
    "pixelart":"pixelart",
    "colorpencil":["pencil","pass"],
    "pixelpencil":["pixelart","pencil"],
}

app.engine('handlebars', exphbs({defaultLayout: 'main'}));
app.set('view engine', 'handlebars');

app.get('/', function (req, res) {
    res.render('home',{
        filters: registeredFilters
    });
});

app.get('/home', function (req, res) {
    res.render('home',{
        filters: registeredFilters
    });
});

app.use('/uploadedimages', express.static('uploadedimages'));
app.use(express.static(pathtools.join(__dirname, '/public')));
//app.use('/views',express.static('views'));

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
        console.log('fields: ' + JSON.stringify(fields));
        //console.log('files: ' + JSON.stringify(files.filetoupload));
        var selectedFilter = fields['selectedfilter'];
        
        var selectedFilters = filterMappingDict[selectedFilter]
        if(selectedFilters){

            var comment = fields.comment;
            var imageFile = files.filetoupload;
            if (imageFile){
                var name = imageFile.name;
                var path = imageFile.path;
                var type = imageFile.type;

                if(path){
                    var pathmeta = utilfns.getPathMeta(name);
                    var destname = pathmeta.name + "-dst" + pathmeta.ext;
                    var destpath = imgsuploaddir + "/" + destname;

                    
                    //filtermanager.callFileProcFunctionByName(selectedFilter,path,destpath)
                    filtermanager.compose(selectedFilters,path,destpath)
                    .then((grownpath)=>{
                        var imagesrclink = "uploadedimages/" + name;
                        var imagedestlink = "uploadedimages/" + destname;
                        console.log("Redirecting client to image: " + imagedestlink)
                        res.redirect("/home?imgsrc=" + imagesrclink + "&imgdst=" + imagedestlink);
                    })
                    .catch((err)=>{
                        console.log("Error occurend when trying to call filter: " + err)
                    });

                    /*if(selectedFilter == "pixelart"){
                        bincaller.regiongrow(path,destpath)
                        .then((grownpath)=>{
                            let imagesrclink = "uploadedimages/" + name;
                            let imagedestlink = "uploadedimages/" + destname;
                            res.redirect("/home?imgsrc=" + imagesrclink + "&imgdst=" + imagedestlink);
                        })
                        .catch((err)=>{
                            console.log("Error occurend when executing external binary: " + err)
                        });
                    }*/


                    /*console.log("spawning Python script")
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
                    });*/
                }
            } else {
                res.send(404);
            }
        }
        else{
            console.log("user sent invalid filter")
            res.send("Invalid Filter");
        }

    });
});
 
server.listen(3000, function () {
    console.log('express-handlebars example server listening on: 3000');
});