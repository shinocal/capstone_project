
%%
Var=VarName11;
[postal_codes] = find_postal_codes(Var,postal_codes);

%%
for i =1:length(postal_codes)
    if isempty(postal_codes{i})==1
        display(strcat(num2str(i),' is empty'))
    end
end

%%
% states=cell(length(stateAZ),1);
Var=VarName12;
[states] = find_states(Var,states);

%%
for i =1:length(states)
    if isempty(states{i})==1
        display(strcat(num2str(i),' is empty'))
    end
end

%%
% categories=cell(length(stateAZ),1);
for i =1:18
Var=MAT2(:,i);
[categories] = find_categories(Var,categories);
end

%%
for i =1:length(categories)
    if isempty(categories{i})==1
        display(strcat(num2str(i),' is empty'))
    end
end

%%
businessID=cell(length(categories),1);
for i =1:length(VarName1)
    businessID{i}=VarName1{i}(17:length(VarName1{i})-1);
end

%%

businessID_rev=cell(length(categories),1);
for i =1:length(businessID_revlong)
    businessID_rev{i}=businessID_revlong{i}(14:length(businessID_revlong{i})-1);
end

%%
clear C inds busID_FF
C{1}='Fast Food';
C{2}='Pizza';
C{3}='Burgers';
C{4}='Pretzels';
C{5}='Chicken Wings';
C{6}='Chicken Shop';
C{7}='Comfort Food';
C{8}='Diners';
C{9}='Donuts';
C{10}='Fish & Chips';
C{11}='Food Court';
C{12}='Food Stands';
C{13}='Food Trucks';
C{14}='Gastropubs';
C{15}='Sandwiches';
C{16}='Tacos';
C{17}='Ice Cream & Frozen Yogurt';

p=1;

for i =1:length(categories)
clear TF
TF= strcmpi(categories{i},C);  
if sum(TF)>0
inds(p)=i;
p=p+1;
end    
end

%%

busID_FF=businessID(inds);
states_FF=states(inds);
[C,IA,IB] = intersect(busID_FF,businessID_rev);
ratings_ff=rating(IB);
busID_FF=busID_FF(IA);
states_FF=states_FF(IA);
[allstates,~,IC] = unique(states_FF);

%%
clear state_ratings ll ul av
p=1;
for i =1:length(allstates)

    if length(find(IC==i))>=30   
    states_incl{p}=allstates{i};
    state_ratings{p}=ratings_ff(IC==i);
    [ll(p),ul(p),av(p)] = sem(state_ratings{p});  
    p=p+1;
    end

end
%%
Y=av;
L=av-ll;
U=ul-av;

figure
bar(Y)
hold on
errorbar(1:11,Y,L,U,'k.')
hold off
ylim([2.5 4])
set(gca,'xticklabel',states_incl)
xlabel('State Code')
ylabel('Mean Rating +/- S.E.M.')
title('Average Fast Food Yelp Ratings by State')


%%
diab_perc=PCT_DIABETES_ADULTS10;
FFperc=FFRPTH07;
X=find(isnan(diab_perc)==1);
diab_perc(X)=[];
FFperc(X)=[];
%%
X=find(isnan(FFperc)==1);
diab_perc(X)=[];
FFperc(X)=[];
%%

[rho,pval]=corr(diab_perc,FFperc)

%%
figure
plot(diab_perc,FFperc,'k.')

%%
Y=cat(2,diab_perc,FFperc);

corrplot(Y)

%%

[B,DEV,STATS] = glmfit(FFperc,diab_perc);

[YHAT,DYLO,DYHI] = glmval(B,FFperc,'identity',STATS);

close all
figure
plot(FFperc,diab_perc,'k.')
hold on
errorbar(FFperc,YHAT,DYLO,DYHI,'b')
hold off
ylim([0 20])
% xlim([0 5])
xlabel('Fast Food / 1000 pop.')
ylabel('% diabetes')
title('GLM fit w %95 C.I.')
%%  http://www.unitedstateszipcodes.org/zip-code-database/
county(1)=[];
zip1(1)=[];
state(1)=[];
type1(1)=[];
%%
p=1;
for i =1:length(type1)
%  if strcmpi(type1{i},'UNIQUE')==1 || strcmpi(type1{i},'PO BOX')==1 ||...
%     strcmpi(state{i},'PR')==1 || strcmpi(state{i},'VI')==1     

 if strcmpi(type1{i},'PO BOX')==1 ||...
    strcmpi(state{i},'PR')==1 || strcmpi(state{i},'VI')==1  
    rem_el(p)=i;
    p=p+1;
 end
end
%%

county(rem_el)=[];
zip1(rem_el)=[];
state(rem_el)=[];
%%
p=1;
for i =1:length(county)
    
    if isempty(county{i})==1 || strcmpi(county{i}(length(county{i})-5:length(county{i})),...
            'County')==0
    countyend(p)=i;    
    p=p+1;
    end
    
end
%%

weird_names=county(countyend);



county(countyend)=[];
zip1(countyend)=[];
state(countyend)=[];
%%

for i=1:length(county)   
    county_trunc{i,1}=county{i}(1:length(county{i})-6);
end
    

%% keep american states from yelp data set

amer_states{1}='AZ';
amer_states{2}='IL';
amer_states{3}='NC';
amer_states{4}='NV';
amer_states{5}='NY';
amer_states{6}='OH';
amer_states{7}='PA';
amer_states{8}='SC';
amer_states{9}='VT';
amer_states{10}='WI';


[C,~,IC] = unique(states);

%%
[C_intersect,IA,IB] = intersect(amer_states,C);
%%
for i =1:length(IB)
    state_ind{i}=find(IC==IB(i));
    num_rev(i)=length(find(IC==IB(i)));
       
end


US_ind=cell2mat(state_ind');

US_postal_codes_char=postal_codes(US_ind);

%%

for i =1:length(US_postal_codes_char)
   
    if isempty(str2num(US_postal_codes_char{i}))==0
    US_postcodes(i)=str2num(US_postal_codes_char{i});
    end
end


%%
[C_postcodes,~,IC_postcodes] = unique(US_postcodes);
[C,IA] = setdiff(C_postcodes,zip1);
C_postcodes(IA)=[];

for i =1:length(C_postcodes)
    
zip_ind(i)=find(zip1==C_postcodes(i));    
    
end


%%

for i =1:length(county_trunc)

    
p=1;
while p==1;
if strcmpi(county_trunc{i}(length(county_trunc{i})),' ')==1
    county_trunc{i}=county_trunc{i}(1:length(county_trunc{i})-1);
    p=1;
end

if strcmpi(county_trunc{i}(length(county_trunc{i})),' ')==0
    p=0;
end
end
    
end

%%
cnty_crss=county_trunc(zip_ind);
C = setdiff(cnty_crss,CountyName);
state_crss=state(zip_ind);
state_crsref=state;

%%
for i =1:length(cnty_crss)
    for n=1:length(CountyName)
    if strcmpi(CountyName{n},cnty_crss{i})==1 && ...
            strcmpi(State{n},state_crss{i})==1
    FDA_refind(i)=n;    
    end
    end
end
%%

for i =1:length(C_postcodes)  
yelp_postcodes_refind{i}= find(US_postcodes==C_postcodes(i));
num_rev_zipcode(i)=length(yelp_postcodes_refind{i});
end

%%
figure
X=1:1:3000;
hist(num_rev_zipcode,X)
xlabel('# of Yelp businesses')
ylabel('zip codes [count]')

%%


FDA_counties=CountyName(FDA_refind);
FDA_states=State(FDA_refind);
FDA_merged=strcat(FDA_counties,FDA_states);
[C_FDA,~,IC_FDA] = unique(FDA_merged);

for i =1:length(C_FDA)
yelp_counties_refind{i}=cell2mat(yelp_postcodes_refind(find(IC_FDA==i)));
num_rev_county(i)=length(yelp_counties_refind{i});   
end


%%
figure
X=1:100:50000;
hist(num_rev_county,X)
xlabel('# of Yelp businesses')
ylabel('counties [count]')
%%

[Y,I]=sort(num_rev_county);

%%

US_bus_ID=businessID(US_ind);
[US_busID_FF,IA,IB] = intersect(US_bus_ID,busID_FF);

%%

for i=1:length(yelp_counties_refind)
county_busID{i}=US_bus_ID(yelp_counties_refind{i});
county_busID_FF{i}=intersect(county_busID{i},busID_FF);  
num_ff_county(i)=length(county_busID_FF{i});
end

%%

[C_busID_rev,IA_busID_rev,IC_busID_rev] = unique(businessID_rev);
LIA=ismember(C_busID_rev,US_busID_FF);
C_busID_rev_FF=C_busID_rev(LIA);

%%
FF_inds_rev=find(LIA);

for i =1:length(county_busID_FF) 
for n=1:length(county_busID_FF{i})
for q=1:length(C_busID_rev_FF)
if strcmpi(county_busID_FF{i}{n},C_busID_rev_FF{q})==1 
rev_inds{i}{n}=find(IC_busID_rev==FF_inds_rev(q));
end
end
end
end

%%

for i =1:length(rev_inds)  
    county_FF_rtngs{i}=rating(cell2mat(rev_inds{i}'));   
end

%%

[~,IA_FDA,~] = unique(FDA_merged);
FDA_county_ind=FDA_refind(IA_FDA);
%%

[Y,I]=sort(FIPS_RESTAURANTS);
FFRPTH2=FFRPTH2(I);

%%
[Y,I]=sort(FIPS_HEALTH);
PCT_OBESE_ADULTS10=PCT_OBESE_ADULTS10(I);

%%
PCT_DIABETES_ADULTS1=PCT_DIABETES_ADULTS1(I);

%%
diab_perc_yelp=PCT_DIABETES_ADULTS1(FDA_county_ind);

%%
FFperc_yelp=FFRPTH2(FDA_county_ind);

%%

for i =1:length(county_FF_rtngs)
num_ffratings_cnty(i)=length(county_FF_rtngs{i});    
av_ffrat_cnty(i)=mean(county_FF_rtngs{i});
var_ffrat_cnty(i)=std(county_FF_rtngs{i})/sqrt(num_ffratings_cnty(i));

lowrat(i)=(length(find(county_FF_rtngs{i}==1))...
    /num_ffratings_cnty(i))*100;

end

%%
close all
clc

min_rev=100;
number_reviews_included=length(find(num_ffratings_cnty>min_rev))
use_inds=find(num_ffratings_cnty>min_rev);
use_diabperc=diab_perc_yelp(use_inds);
use_FFperc=FFperc_yelp(use_inds);
use_obesert=PCT_OBESE_ADULTS10(use_inds);
use_avFFrat=av_ffrat_cnty(use_inds)';
use_numffrat=num_ffratings_cnty(use_inds)';
use_counties=C_FDA(use_inds);
use_varFFrat=var_ffrat_cnty(use_inds)';
use_lowrat=lowrat(use_inds)';

X=[use_lowrat];
Y=use_diabperc;

[B,DEV,STATS] = glmfit(X,Y);
[YHAT,DYLO,DYHI] = glmval(B,X,'identity',STATS);
beta=STATS.beta(2)
p=STATS.p(2)

figure
plot(X(:,1),Y,'k.')
hold on
errorbar(X(:,1),YHAT,DYLO,DYHI,'b')
hold off
xlabel('Percentage Rating of 1')
ylabel('% Diabetes')
title('GLM fit w %95 C.I.')

%%

use_cnty_FFrat=county_FF_rtngs(use_inds);
response=[];
predictor1=[];
predictor2=[];
predictor3=[];
predictor4=[];
predictor5=[];
for i =1:length(use_numffrat)
response=cat(1,response,(ones(1,use_numffrat(i))*use_diabperc(i))');  
% response=cat(1,response,(ones(1,use_numffrat(i))*use_FFperc(i))'); 
predictor1=cat(1,predictor1,use_cnty_FFrat{i}==1); 
predictor2=cat(1,predictor2,use_cnty_FFrat{i}==2); 
predictor3=cat(1,predictor3,use_cnty_FFrat{i}==3); 
predictor4=cat(1,predictor4,use_cnty_FFrat{i}==4); 
predictor5=cat(1,predictor5,use_cnty_FFrat{i}==5); 
% predictor2=cat(1,predictor2,(ones(1,use_numffrat(i))*use_FFperc(i))');  
end

%%

X=[predictor1,predictor2,predictor3,predictor4,predictor5];
Y=response;

[B,DEV,STATS] = glmfit(X,Y);
[YHAT,DYLO,DYHI] = glmval(B,X,'identity',STATS);
beta=STATS.beta
p=STATS.p

% figure
% plot(X(:,1),Y,'k.')
% hold on
% errorbar(X(:,1),YHAT,DYLO,DYHI,'b')
% hold off

%%
clear ll ul av
close all
for i=1:5
[ll(i),ul(i),av(i)] = sem(response(X(:,i)==1)); 

end

Y=av;
L=av-ll;
U=ul-av;

figure
subplot(2,1,1)
bar(Y)
hold on
errorbar(1:5,Y,L,U,'k.')
plot([0,6],[mean(response) mean(response)],'r--')
hold off
ylim([8.42 8.48])
ylabel('Diabetes Likelihood [%, +/- S.E.M.]')
title('Diabetes Likelihood of Yelp Reviewers')
subplot(2,1,2)
bar(beta(2:6))
hold on
Y=beta(2:6);
L=STATS.se(2:6);
U=STATS.se(2:6);
errorbar(1:5,Y,L,U,'k.')
hold on
plot([1],[-0.028],'k*')
hold off
ylabel('Beta +/- S.E.')
xlabel('Rating [1-5 stars]')


