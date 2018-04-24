
const PORT = process.env.PORT || 3000
const IMAGE_FIELD_NAME = 'image'
const IMAGE_EXT_ALLOWED = ['jpg', 'png']
const FILE_UPLOAD_DIR = process.env.FILE_UPLOAD_DIR || '/tmp/epsiOCR'

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
	app.use(fileUpload())

	app.get('/', (req, res) => {
	  res.sendFile(path.join(__dirname, '/views/index.html'))
	})

	app.post('/predict', (req, res) => {
		const image = (req.files || {})[IMAGE_FIELD_NAME]
		if (!image || !imageIsValid(image.name)){
			return res.status(400).send(`No files were uploaded or the file don't have any of the following extentions: ${IMAGE_EXT_ALLOWED.join(', ')}`)
		}
		else{
			const imageUuid = uuid()
			const newImagePath = path.join(FILE_UPLOAD_DIR, `${imageUuid}.${getImageExtention(image.name)}`)
			image.mv(newImagePath)
			.then(() => {
				console.log(newImagePath)
				const pythonProcess = spawn('python',[path.join(__dirname, '../evaluate/train.py'), newImagePath], {env: {
					MODEL_PATH: path.join(__dirname, '../evaluate/model')
				}})
				
				pythonProcess.stdout.on('data', (data) => {
				  res.write(data)
				})

				pythonProcess.on('close', (code) => {
				  res.end()
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