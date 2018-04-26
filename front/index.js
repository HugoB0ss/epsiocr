
const PORT = process.env.PORT || 3000
const IMAGE_FIELD_NAME = 'image'
const IMAGE_EXT_ALLOWED = ['jpg', 'png']
const FILE_UPLOAD_DIR = process.env.FILE_UPLOAD_DIR || '/tmp/epsiOCR'
const EOL = require('os').EOL

const getImageExtention = (image) => image.split('.').pop()
const imageIsValid = (image) => IMAGE_EXT_ALLOWED.includes(getImageExtention(image))

const spawn = require("child_process").spawn;

const path = require('path')
const mkdirp = require('mkdirp')
const uuid = require('uuid/v4')
const express = require('express')
const fileUpload = require('express-fileupload')

mkdirp(FILE_UPLOAD_DIR, (err) => {
	if(err){
		console.error(`Can't create ${FILE_UPLOAD_DIR} directory !`)
		console.error(err)
		process.exit(1)
	}
	
	const app = express()
	app.use(fileUpload({limits: { fileSize: 4 * 1024 * 1024 }}))
	app.use('/bootstrap', express.static(path.resolve('node_modules/bootstrap/dist')))
	app.use('/jquery', express.static(path.resolve('node_modules/jquery/dist')))
	app.use('/style', express.static(path.resolve('style')))
	app.get('/', (req, res) => {
	  res.sendFile(path.join(__dirname, '/views/index.html'))
	})

	app.post('/predict', (req, res) => {
		let images = (req.files || {})[IMAGE_FIELD_NAME]
		images = images && !Array.isArray(images) ? [images] : images
		if (!images || images.some( (image) => !imageIsValid(image.name))){
			return res.status(400).send(`No files were uploaded or one of the files don't have any of the following extentions: ${IMAGE_EXT_ALLOWED.join(', ')}`)
		}
		else{
			let newImagesPath = []
			Promise.all(images.map((image) => {
				const imageUuid = uuid()
				const newImagePath = path.normalize(path.join(FILE_UPLOAD_DIR, `${imageUuid}.${getImageExtention(image.name)}`))
				newImagesPath = [...newImagesPath, [image.name, newImagePath]]
				return image.mv(newImagePath)				
			}))
			.then(() => {
				const pythonProcess = spawn('python',[path.join(__dirname, '../evaluate/train.py'), ...newImagesPath.map((p) => p[1])], {env: {
					MODEL_PATH: path.join(__dirname, '../evaluate/model')
				}})
				
				let pythonData = []
				pythonProcess.stdout.on('data', (data) => {
				  data = data.toString().split(EOL)
				  data.pop()
				  data = data.reduce((acc, curr) => {
					  const [path, result] = curr.split('|')
					  const originalPath = newImagesPath.find((p) => p[1] === path)[0]
					  console.log(originalPath, result)
					  return {...acc, [originalPath]: result}
				  }, {})
				  res.send(data)
				})
			})
			.catch((err) => {
				console.error(err)
				res.status(500).send(`Error during moving the uploded file`)
			})
		}
	})

	app.listen(PORT, () => {
		console.log(`EpsiOCR started on port ${PORT} !`)
	})

})