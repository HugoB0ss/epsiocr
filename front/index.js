// variables d'environnement
const PORT = process.env.PORT || 3000
const IMAGE_FIELD_NAME = 'image'
const IMAGE_EXT_ALLOWED = ['jpg', 'png']
const FILE_UPLOAD_DIR = process.env.FILE_UPLOAD_DIR || '/tmp/epsiOCR'
const EOL = require('os').EOL

// Fonctions
const getImageExtention = (image) => image.split('.').pop()
const imageIsValid = (image) => IMAGE_EXT_ALLOWED.includes(getImageExtention(image))
const getTime = () => (new Date()).getTime()

const spawn = require('child_process').spawn

// Modules
const path = require('path')
const mkdirp = require('mkdirp')
const uuid = require('uuid/v4')
const express = require('express')
const fileUpload = require('express-fileupload')

mkdirp(FILE_UPLOAD_DIR, (err) => {
  // On crée le dossier temporaire si nécessaire
  if (err) {
    console.error(`Can't create ${FILE_UPLOAD_DIR} directory !`)
    console.error(err)
    process.exit(1)
  }

  // On créee un serveur web tout en limitant la taille d'upload maximale
  const app = express()
  app.use(fileUpload({limits: { fileSize: 4 * 1024 * 1024 }}))
  app.use('/bootstrap', express.static(path.resolve('node_modules/bootstrap/dist')))
  app.use('/jquery', express.static(path.resolve('node_modules/jquery/dist')))
  app.use('/style', express.static(path.resolve('style')))

  // Page d'acceuil
  app.get('/', (req, res) => {
	  res.sendFile(path.join(__dirname, '/views/index.html'))
  })

  // Module d'intelligence
  app.post('/predict', (req, res) => {
    let images = (req.files || {})[IMAGE_FIELD_NAME]
    images = images && !Array.isArray(images) ? [images] : images

    // On vérifie que le fichier possède un format utilisable
    if (!images || images.some((image) => !imageIsValid(image.name))) {
      return res.status(400).send(`No files were uploaded or one of the files don't have any of the following extentions: ${IMAGE_EXT_ALLOWED.join(', ')}`)
    } else {
      let newImagesPath = []

      // On déplace chaque Image dans le dossier temporaire, tout en enrengistrant leur nom d'origine
      Promise.all(images.map((image) => {
        const imageUuid = uuid()
        const newImagePath = path.normalize(path.join(FILE_UPLOAD_DIR, `${imageUuid}.${getImageExtention(image.name)}`))
        newImagesPath = [...newImagesPath, [image.name, newImagePath]]
        return image.mv(newImagePath)
      }))
        .then(() => {
          // On crée un processus python pour découper nos images envoyées avec le temps d'exécution
          // console.log(newImagesPath)

          const startTime = getTime()
          const pythonProcess = spawn('python', [path.join(__dirname, '../evaluate/split.py'), ...newImagesPath.map((p) => p[1])], {env: {
            MODEL_PATH: path.join(__dirname, '../evaluate/model')
          }})
          pythonProcess.stdout.on('data', (data) => {
            data = data.toString().split(EOL)
            data.pop()
            newData = []
            var i, j, temparray, chunk = 5

            // On regroupe les données
            for (i = 0, j = data.length; i < j; i += chunk) {
              temparray = data.slice(i, i + chunk)
              newData = [...newData, temparray]
            }

            /*	console.log(data[0])
					data.pop()
				  data = data.reduce((acc, curr) => {
					  const [path, result] = curr.split('|')
					  const originalPath = newImagesPath.find((p) => p[1] === path)[0]
					  console.log("LOG")
					  console.log(originalPath, result)
					  return {...acc, [originalPath]: result}
				  }, {}) */

				  // On présente les données au format JSON
				  console.log(newData)
				  data = newData.reduce((a, c, i) => ({...a, [newImagesPath[i][0]]: c}), {})

				  const diff_time = getTime() - startTime
				  res.send({data, _execution_time: `${diff_time / 1000} seconds`})
          })

          // Si on a une erreur, alors on arrette le process et on affiche le script
          pythonProcess.stderr.on('data', (data) => {
            pythonProcess.kill('SIGINT')
					  data = data.toString().split(EOL)
					  data.pop()
					  console.log('ERROR')
					  console.log(data)
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
