#!/usr/bin/python2.7


import xml.etree.ElementTree as ET
import pdb

"""
把数学表达式的xml文件解释为字符串返回
usage:
	object = Formula(xmlFileName)
	result = object.get_expection()
"""
class Formula:
	__symbol__dict = {"pi": "\pi ", "sum": "\sum "}  # 数学标识符字典

	# 构造函数，文件名做参数
	def __init__(self, xmlFileName):
		self.__tree = ET.parse(xmlFileName)
		self.__items = self.__tree.getroot()
		self.__original_result = ""  # 保存结果的字符串 可能含有sub_formal
		self.__result = ""  # 最终的结果
		self.__delimiter = ","
		

		# 标签属性
		self.__letter = "letter"
		self.__oper_or_num = "oper_or_num"
		self.__up = "up"
		self.__down = "down"
		self.__sub_formula = "sub_formula"

	# 写入文件，生成tex
	def write_tex_file(self, target='texFile.tex'):
		fo_head = open("../resource/head.tex", "r")
		fo_tail = open("../resource/tail.tex", "r")
		fw = open(target, "w")
		for line in fo_head.readlines():
			fw.write(line)

		fw.write(self.__result)

		for line in fo_tail.readlines():
			fw.write(line)

		fw.close()
		fo_tail.close()
		fo_head.close()


	# 标签属性是letter, 获取letter添加到string
	def __is_letter(self, item):
		string = ""
		for it in item.findall(".//*"):
			if it.text != None:
				if it.tag == self.__down:
					it.text = "_{" + it.text + "}"
				if it.tag == self.__up:
					it.text = "^{" + it.text + "}"
				if it.text in Formula.__symbol__dict:
				# if Formula.__symbol__dict.has_key(it.text):
					string += Formula.__symbol__dict[it.text]
				else:
					string += it.text
		return string

	def __is_operator_and_num(self, item):
		string = ""
		for it in item.findall(self.__oper_or_num):
			string += it.text
		return string

	# result和列表合并
	def __merge(self):
		sub_formula_list = self.__original_result.split(self.__delimiter)[1:]
		tmp_result = self.__original_result.split(self.__delimiter)[0]

		for sub_formula in sub_formula_list:
			if tmp_result.find(self.__sub_formula):
				tmp_result = tmp_result.replace(self.__sub_formula,
					sub_formula, 1)
		tmp_result = '$$' + tmp_result + '$$'
		return tmp_result

	def expection(self):
		self.__original_result = self.get_expection(self.__items)
		self.__result = self.__merge()
		return self.__result

	def get_expection(self, items):
		string = ""
		for item in items:
			s = item.attrib
			l = list(s.values())
			if (l[0] == self.__letter):
				string += Formula.__is_letter(self, item)
			elif (l[0] == self.__oper_or_num):
				string += Formula.__is_operator_and_num(self, item)
			elif (l[0] == self.__sub_formula):
				string += self.__delimiter
				string += self.get_expection(item)

		return string

# if __name__ == '__main__':
# 	object = Formula("math3.xml")
# 	print(object.expection())
# 	object.write_tex_file()
