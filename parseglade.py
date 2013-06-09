f=open("glade.glade","r")

s=f.read()
f.close()
def fixnegative(x):
    global s
    if x==-1:
        return len(s)
    else:
        return x

#find <child> and </object>
pos=0
while True:
    try:
        p1=s.index("GtkBox",pos)
        p2=s.find("<child>",p1)
        #print p1
        #print p2
        p3=s.find("</object>",p1)
        #print p3
        p2=fixnegative(p2)
        p3=fixnegative(p3)
        if p2<p3:
            use=p2
        else:
            use=p3
        #print s[p1:use]
        #print "----"
        if '<property name="orientation">vertical</property>' in s[p1:use]:
            s=s[0:p1+3]+"V"+s[p1+3:]
        else:
            s=s[0:p1+3]+"H"+s[p1+3:]

        pos=p1+12
        
    except:
        break

s=s.replace("GtkGrid", "GtkTable")

f=open("gladeb.glade","w")
f.write(s)
f.close()
