function nocache(module) { require("fs").watchFile(require("path").resolve(module), () => { delete require.cache[require.resolve(module)] }) }
nocache("./index.node");

process.env['RUST_BACKTRACE'] = 1;

module.exports.runtime = require("./index.node")
