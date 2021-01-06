dataset = pd.read_csv("C:/Users/Gerard/Desktop/TFM/Covid Dataset.csv")

#Creem tantes llistes buides com columnes

a = list(range(5434))
b = list(range(5434))
c = list(range(5434))
d = list(range(5434))
e = list(range(5434))
f = list(range(5434))
g = list(range(5434))
h = list(range(5434))
h1 = list(range(5434))
h2 = list(range(5434))
k = list(range(5434))
l = list(range(5434))
m = list(range(5434))
n = list(range(5434))
o = list(range(5434))
p = list(range(5434))
q = list(range(5434))
r = list(range(5434))
s = list(range(5434))
t = list(range(5434))
u = list(range(5434))

#Substituim els valors Yes No per 1 o 0

j = 0
for i in dataset.iloc[:,0]:

    if i == 'Yes':
        a[j] = '1'
        j = j+1
    else:
        a[j] = '0'
        j = j+1

j = 0
for i in dataset.iloc[:,1]:

    if i == 'Yes':
        b[j] = '1'
        j = j+1
    else:
        b[j] = '0'
        j = j+1

j = 0
for i in dataset.iloc[:,2]:

    if i == 'Yes':
        c[j] = '1'
        j = j+1
    else:
        c[j] = '0'
        j = j+1

j = 0
for i in dataset.iloc[:,3]:

    if i == 'Yes':
        d[j] = '1'
        j = j+1
    else:
        d[j] = '0'
        j = j+1

j = 0
for i in dataset.iloc[:,4]:

    if i == 'Yes':
        e[j] = '1'
        j = j+1
    else:
        e[j] = '0'
        j = j+1

j = 0
for i in dataset.iloc[:,5]:

    if i == 'Yes':
        f[j] = '1'
        j = j+1
    else:
        f[j] = '0'
        j = j+1

j = 0
for i in dataset.iloc[:,6]:

    if i == 'Yes':
        g[j] = '1'
        j = j+1
    else:
        g[j] = '0'
        j = j+1

j = 0
for i in dataset.iloc[:,7]:

    if i == 'Yes':
        h[j] = '1'
        j = j+1
    else:
        h[j] = '0'
        j = j+1


j = 0
for i in dataset.iloc[:,8]:

    if i == 'Yes':
        h1[j] = '1'
        j = j+1
    else:
        h1[j] = '0'
        j = j+1

j = 0
for i in dataset.iloc[:,9]:

    if i == 'Yes':
        h2[j] = '1'
        j = j+1
    else:
        h2[j] = '0'
        j = j+1

j = 0
for i in dataset.iloc[:,10]:

    if i == 'Yes':
        k[j] = '1'
        j = j+1
    else:
        k[j] = '0'
        j = j+1

j = 0
for i in dataset.iloc[:,11]:

    if i == 'Yes':
        l[j] = '1'
        j = j+1
    else:
        l[j] = '0'
        j = j+1

j = 0
for i in dataset.iloc[:,12]:

    if i == 'Yes':
        m[j] = '1'
        j = j+1
    else:
        m[j] = '0'
        j = j+1

j = 0
for i in dataset.iloc[:,13]:

    if i == 'Yes':
        n[j] = '1'
        j = j+1
    else:
        n[j] = '0'
        j = j+1

j = 0
for i in dataset.iloc[:,14]:

    if i == 'Yes':
        o[j] = '1'
        j = j+1
    else:
        o[j] = '0'
        j = j+1

j = 0
for i in dataset.iloc[:,15]:

    if i == 'Yes':
        p[j] = '1'
        j = j+1
    else:
        p[j] = '0'
        j = j+1

j = 0
for i in dataset.iloc[:,16]:

    if i == 'Yes':
        q[j] = '1'
        j = j+1
    else:
        q[j] = '0'
        j = j+1

j = 0
for i in dataset.iloc[:,17]:

    if i == 'Yes':
        r[j] = '1'
        j = j+1
    else:
        r[j] = '0'
        j = j+1

j = 0
for i in dataset.iloc[:,18]:

    if i == 'Yes':
        s[j] = '1'
        j = j+1
    else:
        s[j] = '0'
        j = j+1

j = 0
for i in dataset.iloc[:,19]:

    if i == 'Yes':
        t[j] = '1'
        j = j+1
    else:
        t[j] = '0'
        j = j+1

j = 0
for i in dataset.iloc[:,20]:

    if i == 'Yes':
        u[j] = '1'
        j = j+1
    else:
        u[j] = '0'
        j = j+1


#Creem el dataframe buit

columnes = ['Breathing Problem', 'Fever', 'Dry Cough', 'Sore throat',
       'Running Nose', 'Asthma', 'Chronic Lung Disease', 'Headache',
       'Heart Disease', 'Diabetes', 'Hyper Tension', 'Fatigue ',
       'Gastrointestinal ', 'Abroad travel', 'Contact with COVID Patient',
       'Attended Large Gathering', 'Visited Public Exposed Places',
       'Family working in Public Exposed Places', 'Wearing Masks',
       'Sanitization from Market', 'COVID-19']

#Li entrem les dades

df = pd.DataFrame(np.arange(114114).reshape(5434,21), columns= columnes)
df.iloc[:,0] = a
df.iloc[:,1] = b
df.iloc[:,2] = c
df.iloc[:,3] = d
df.iloc[:,4] = e
df.iloc[:,5] = f
df.iloc[:,6] = g
df.iloc[:,7] = h
df.iloc[:,8] = h1
df.iloc[:,9] = h2
df.iloc[:,10] = k
df.iloc[:,11] = l
df.iloc[:,12] = m
df.iloc[:,13] = n
df.iloc[:,14] = o
df.iloc[:,15] = p
df.iloc[:,16] = q
df.iloc[:,17] = r
df.iloc[:,18] = s
df.iloc[:,19] = t
df.iloc[:,20] = u

#Canviem el tipus dels valors

dataset = df
print(dataset.columns)

dataset['Breathing Problem'] = dataset['Breathing Problem'].astype('float64')
dataset['Fever'] = dataset['Fever'].astype('float64')
dataset['Dry Cough'] = dataset['Dry Cough'].astype('float64')
dataset['Sore throat'] = dataset['Sore throat'].astype('float64')
dataset['Running Nose'] = dataset['Running Nose'].astype('float64')
dataset['Asthma'] = dataset['Asthma'].astype('float64')
dataset['Chronic Lung Disease'] = dataset['Chronic Lung Disease'].astype('float64')
dataset['Headache'] = dataset['Headache'].astype('float64')
dataset['Heart Disease'] = dataset['Heart Disease'].astype('float64')
dataset['Diabetes'] = dataset['Diabetes'].astype('float64')
dataset['Hyper Tension'] = dataset['Hyper Tension'].astype('float64')
dataset['Fatigue '] = dataset['Fatigue '].astype('float64')
dataset['Gastrointestinal '] = dataset['Gastrointestinal '].astype('float64')
dataset['Abroad travel'] = dataset['Abroad travel'].astype('float64')
dataset['Contact with COVID Patient'] = dataset['Contact with COVID Patient'].astype('float64')
dataset['Attended Large Gathering'] = dataset['Attended Large Gathering'].astype('float64')
dataset['Visited Public Exposed Places'] = dataset['Visited Public Exposed Places'].astype('float64')
dataset['Family working in Public Exposed Places'] = dataset['Family working in Public Exposed Places'].astype('float64')
dataset['Wearing Masks'] = dataset['Wearing Masks'].astype('float64')
dataset['Sanitization from Market'] = dataset['Sanitization from Market'].astype('float64')
dataset['COVID-19'] = dataset['COVID-19'].astype('float64')

print(dataset.dtypes)

df = dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,20]]
dataset = df
