n1 = float(input("Digite a primeira nota: "))
n2 = float(input("Digite a segunda nota: "))

faltas = int(input("Digite o número de faltas: "))

media = (n1 + n2) / 2

if (media >= 7 and faltas < 10):
    print(f"Aluno aprovado com média {media:.2f} e {faltas} faltas.")

elif (media >= 5 and faltas < 10):

    print(f"Recuperação, nota: {media:.2f} e {faltas} faltas.")

else :
    print(f"Aluno reprovado com média {media:.2f} e {faltas} faltas.")