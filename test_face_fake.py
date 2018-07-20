# codeing:utf-8

import face_fake as ff

fc = ff.FaceChanger()
fc.load_images('bush.jpg', 'trump2.jpg')
fc.run(showProcedure=True)
