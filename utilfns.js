const pathtools = require('path');
function getPathMeta(path){
    var parsedPath = pathtools.parse(path)
    return parsedPath;
}

function getNewPath(srcPath,suffix){
    var parsedPath = pathtools.parse(srcPath)
    return pathtools.join(parsedPath.dir,parsedPath.name + suffix + parsedPath.ext);
}

module.exports = {
    'getPathMeta':getPathMeta,
    'getNewPath':getNewPath
}