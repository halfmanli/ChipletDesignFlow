# Copyright 2000, David Rhodes and Robert Dick
# All rights reserved.
# Use with GNU make.

CC := g++
TO := -o

C_OPTS := -O0 -ffor-scope -ftemplate-depth-50 \
  -DROB_DEBUG -ggdb -O0 -pedantic -I. -std=c++11

L_OPTS := -lstdc++ -lm

# Debug
C_OPTS += -ggdb -DROB_DEBUG

# Profile
#C_OPTS += -p
#L_OPTS += -p -lc_p

# Optimize
# C_OPTS += -s -O3

COMPILE := $(CC) -c $(OPTS) $(C_OPTS)

LINK := $(CC) $(OPTS) $(L_OPTS)

############################################################################## 
# Rules
############################################################################## 
%.o : %.cc
	@echo "***** Compiling -> $@"
	@$(COMPILE) $(TO) $@ $<

%.d : %.cc
	@echo "***** Depending -> $@"
	@sh -ec '$(DEPENDS) $< | $(SED_DEPEND) > $@'

tgff: ArgPack.o DBase.o PGraph.o TGraph.o TG.o main.o psprint.o RStd.o RString.o RGen.o RMath.o Epsilon.o Interface.o Graph.o
	$(REPORT)
	@echo "***** Linking -> $@"
	@$(LINK) $(TO) $@ $(filter %.o,$^)
	@echo

ArgPack.o: ArgPack.cc RMath.h RVector.h RStd.h RStd.cct Interface.h \
 RFunctional.h Interface.cct RVector.cct Epsilon.h Epsilon.cct RMath.cct \
 RPair.h RPair.cct RecVector.h FVector.h RAlgo.h RAlgo.cct FVector.cct \
 RecVector.cct ArgPack.h PGraph.h TG.h RString.h RString.cct Graph.h \
 Graph.cct TGraph.h RGen.h HolderPtr.h HolderPtr.cct DBase.h
DBase.o: DBase.cc DBase.h RString.h RFunctional.h RString.cct RecVector.h \
 Interface.h RStd.h RStd.cct Interface.cct RVector.h RVector.cct \
 FVector.h RAlgo.h RAlgo.cct FVector.cct RecVector.cct ArgPack.h RGen.h \
 HolderPtr.h HolderPtr.cct Epsilon.h Epsilon.cct RMath.h RMath.cct
Epsilon.o: Epsilon.cc Epsilon.h RFunctional.h Interface.h RStd.h RStd.cct \
 Interface.cct Epsilon.cct
Graph.o: Graph.cc Graph.h RStd.h RStd.cct RVector.h Interface.h \
 RFunctional.h Interface.cct RVector.cct Graph.cct Epsilon.h Epsilon.cct
Interface.o: Interface.cc Interface.h RStd.h RStd.cct RFunctional.h \
 Interface.cct
main.o: main.cc ArgPack.h RVector.h RStd.h RStd.cct Interface.h \
 RFunctional.h Interface.cct RVector.cct
PGraph.o: PGraph.cc PGraph.h RVector.h RStd.h RStd.cct Interface.h \
 RFunctional.h Interface.cct RVector.cct TG.h RString.h RString.cct \
 Graph.h Graph.cct ArgPack.h RGen.h HolderPtr.h HolderPtr.cct RMath.h \
 Epsilon.h Epsilon.cct RMath.cct
psprint.o: psprint.cc RStd.h RStd.cct ArgPack.h RVector.h Interface.h \
 RFunctional.h Interface.cct RVector.cct TGraph.h TG.h RString.h \
 RString.cct Graph.h Graph.cct PGraph.h
RGen.o: RGen.cc RGen.h HolderPtr.h Interface.h RStd.h RStd.cct \
 RFunctional.h Interface.cct HolderPtr.cct RVector.h RVector.cct RMath.h \
 Epsilon.h Epsilon.cct RMath.cct RGenData.h
RMath.o: RMath.cc RMath.h RVector.h RStd.h RStd.cct Interface.h \
 RFunctional.h Interface.cct RVector.cct Epsilon.h Epsilon.cct RMath.cct \
 RAlgo.h RAlgo.cct RGen.h HolderPtr.h HolderPtr.cct RString.h RString.cct
RStd.o: RStd.cc RStd.h RStd.cct
RString.o: RString.cc RString.h RFunctional.h RString.cct RVector.h \
 RStd.h RStd.cct Interface.h Interface.cct RVector.cct RMath.h Epsilon.h \
 Epsilon.cct RMath.cct
TG.o: TG.cc TG.h RString.h RFunctional.h RString.cct RVector.h RStd.h \
 RStd.cct Interface.h Interface.cct RVector.cct Graph.h Graph.cct RMath.h \
 Epsilon.h Epsilon.cct RMath.cct RGen.h HolderPtr.h HolderPtr.cct \
 ArgPack.h
TGraph.o: TGraph.cc RMath.h RVector.h RStd.h RStd.cct Interface.h \
 RFunctional.h Interface.cct RVector.cct Epsilon.h Epsilon.cct RMath.cct \
 TGraph.h TG.h RString.h RString.cct Graph.h Graph.cct ArgPack.h RGen.h \
 HolderPtr.h HolderPtr.cct RAlgo.h RAlgo.cct

