import os

for i in range(0,500000):
	try:
		filename='aeitest_%.12d.dat' % i
		os.remove(filename)
	except Exception as e:
		print('删除失败',e)
	try:
		filename='Outtest_p%.6d.dat' % i
		os.remove(filename)
	except Exception as e:
		print('删除失败',e)
	try:
		filename='Outtest_%.12d.dat' % i
		os.remove(filename)
	except Exception as e:
		print('删除失败',e)
