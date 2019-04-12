const SHA256 = require("crypto-js/sha256");
const fs = require('fs')
const util = require('util')
const fstatpromise = util.promisify(fs.lstat);
const writeFile = util.promisify(fs.writeFile);

var hashPathDict = {}
var hashTableLocation = "hashpathtable.json"

//only exec when table not up to date - can be long operation
function initTableFromDir(dirPath,hashTableLocation){

    hashPathDict = {};

    var stat = fs.lstatSync(dirPath);

    if(stat.isDirectory()){
        fs.readdirSync(dirPath).forEach(file => {
            console.log(file);
            if(fs.lstatSync(file).isFile()){
                var fileContent = fs.readFileSync(file);
                var filehash = SHA256(fileContent);
                hashPathDict[filehash] = file;
            }
        });
    }
    else{
        console.log("Directory: " + dirPath + " does not exist");
    }

    saveTable();
}

function loadTable(){
    hashPathDict = JSON.parse(fs.readFileSync(hashTableLocation));
}

function saveTableSync(){
    fs.writeFileSync(hashTableLocation,JSON.stringify(hashPathDict));
}

function saveTableAsync(){
    return writeFile(hashTableLocation,JSON.stringify(hashPathDict));
}

function addHashToDict(hash,filepath){
    hashPathDict[hash] = filepath;
    saveTable()
}

function getPathFromHash(hash){
    return hashPathDict[hash]
}