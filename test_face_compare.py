# coding:utf-8
import face_compare as fc

face_rec = fc.face_recognition()   # 创建对象
face_rec.inputPerson(name='trump', img_path='/faces/trump.jpg')
# name中写第一个人名字，img_name为图片名字，注意要放在faces文件夹中
vector = face_rec.create128DVectorSpace()  # 提取128维向量，是dlib.vector类的对象
person_data1 = fc.savePersonData(face_rec, vector)
# 将提取出的数据保存到data文件夹，为便于操作返回numpy数组，内容还是一样的

# 导入第二张图片，并提取特征向量
face_rec.inputPerson(name='trump2', img_path='/faces/trump2.jpg')
vector = face_rec.create128DVectorSpace()  # 提取128维向量，是dlib.vector类的对象
person_data2 = fc.savePersonData(face_rec, vector)

# 计算欧式距离，判断是否是同一个人
fc.comparePersonData(person_data1, person_data2)
